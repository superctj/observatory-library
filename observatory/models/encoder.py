import torch

from transformers import (
    AlbertModel,
    AutoTokenizer,
    BertModel,
    RobertaModel,
)

from observatory.models.model_wrapper import ModelWrapper


class BERTFamilyModelWrapper(ModelWrapper):
    """Model wrapper for any BERT-like model whose tokenizer has valid
    attributes `cls_token`, `sep_token`, and `pad_token`.

    To use this class, inherit from it and implement the `get_model` method.
    Override other methods if necessary. Specifically:

    - Override the `get_tokenizer` method if the tokenizer cannot be initiated
    from `AutoTokenizer`.

    - Override the `get_max_input_size` method if the model input size cannot be
    obtained from `model.config.max_position_embeddings`.

    - Override the `get_embedding_dimension` method if the model embedding size
    cannot be obtained from `model.config.hidden_size`.
    """

    def __init__(self, model_name: str, device: torch.device):
        super().__init__(model_name, device)

        # Table serialization and encoding require [CLS] and [SEP] tokens
        assert self.tokenizer.cls_token is not None
        assert self.tokenizer.sep_token is not None

    def get_tokenizer(self) -> AutoTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True
            )
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return tokenizer

    def get_max_input_size(self) -> int:
        return self.model.config.max_position_embeddings

    def get_embedding_dimension(self) -> int:
        return self.model.config.hidden_size

    def infer_embeddings(
        self,
        encoded_inputs: dict,
        cls_positions: list[list[int]] = None,
    ) -> torch.FloatTensor:
        """Infer embeddings from [CLS] tokens.

        Args:
            encoded_inputs:
                A dictionary of encoded inputs.
            cls_positions:
                Positions of [CLS] tokens per table.

        Returns:
            embeddings:
                A tensor of shape (<number of embeddings>, <embedding size>).
        """

        for key in encoded_inputs:
            encoded_inputs[key] = encoded_inputs[key].to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded_inputs)

        batch_last_hidden_state = outputs.last_hidden_state

        if not cls_positions:
            embeddings = batch_last_hidden_state[:, 0, :]
        else:
            embeddings = []

            for i, last_hidden_state in enumerate(batch_last_hidden_state):
                cls_embeddings = torch.stack(
                    [last_hidden_state[pos, :] for pos in cls_positions[i]],
                    dim=0,
                )
                assert cls_embeddings.dim() == 2, (
                    "`cls_embeddings` should be a 2D tensor but has "
                    f"{cls_embeddings.dim()} dimensions."
                )

                embeddings.append(cls_embeddings)

            embeddings = torch.cat(embeddings, dim=0)

        assert embeddings.dim() == 2, (
            "`embeddings` should be a 2D tensor but has "
            f"{cls_embeddings.dim()} dimensions."
        )
        assert embeddings.shape[1] == self.get_embedding_dimension(), (
            "The second dimension  of `embeddings` has an unexpected size of "
            f"{embeddings.shape[1]} while the expected size is "
            f"{self.model.get_embedding_dimension()}."
        )

        return embeddings

    def infer_embeddings_by_averaging_tokens(
        self,
        encoded_inputs: dict,
        span_positions: list[list[tuple[int]]],
    ) -> list[torch.FloatTensor]:
        """Infer embeddings by averaging embeddings of tokens in each span. For
        example, cell embeddings are obtained by averaging embeddings of tokens
        in each cell span.

        Args:
            encoded_inputs:
                A dictionary of encoded inputs.
            span_positions:
                Lists of token spans in each sequence. Each span is represented
                as a tuple of start (inclusive) and end (non-inclusive) indices.

        Returns:
            embeddings_per_table:
                A list of tensors of shape (<number of cells>, <embedding size)
                where each tensor corresponds to a table.
        """

        embeddings_per_table = []

        for key in encoded_inputs:
            encoded_inputs[key] = encoded_inputs[key].to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded_inputs)

        batch_last_hidden_state = outputs.last_hidden_state

        for i, last_hidden_state in enumerate(batch_last_hidden_state):
            embeddings = torch.zeros(
                len(span_positions[i]),
                self.model.config.hidden_size,
                dtype=torch.float,
            )

            for j, span in enumerate(span_positions[i]):
                embeddings[j] = torch.mean(
                    last_hidden_state[span[0] : span[1], :], dim=0
                )

            embeddings_per_table.append(embeddings)

        return embeddings_per_table


class BertModelWrapper(BERTFamilyModelWrapper):
    def get_model(self) -> BertModel:
        try:
            model = BertModel.from_pretrained(
                self.model_name, local_files_only=True
            )
        except OSError:
            model = BertModel.from_pretrained(self.model_name)

        model = model.to(self.device)
        model.eval()

        return model


class RobertaModelWrapper(BERTFamilyModelWrapper):
    def get_model(self) -> RobertaModel:
        try:
            model = RobertaModel.from_pretrained(
                self.model_name, local_files_only=True
            )
        except OSError:
            model = RobertaModel.from_pretrained(self.model_name)

        model = model.to(self.device)
        model.eval()

        return model

    # This method is overridden due to `RuntimeError: CUDA error: device-side
    # assert triggered` occurred when getting the model input size from the
    # model config
    def get_max_input_size(self) -> int:
        return 512


class AlbertModelWrapper(BERTFamilyModelWrapper):
    def get_model(self) -> AlbertModel:
        try:
            model = AlbertModel.from_pretrained(
                self.model_name, local_files_only=True
            )
        except OSError:
            model = AlbertModel.from_pretrained(self.model_name)

        model = model.to(self.device)
        model.eval()

        return model
