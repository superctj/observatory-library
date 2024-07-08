import torch

from transformers import AutoTokenizer, T5Model

from observatory.models.model_wrapper import ModelWrapper


class T5FamilyModelWrapper(ModelWrapper):
    """Model wrapper for any T5-like model."""

    def __init__(self, model_name: str, device: torch.device):
        super().__init__(model_name, device)

    def get_tokenizer(self) -> AutoTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True
            )
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return tokenizer

    def infer_embeddings(
        self,
        encoded_inputs: dict,
        use_decoder_last_hidden_state: bool = True,
        strategy: str = "first",
    ) -> torch.FloatTensor:
        """Infer embeddings from the last hidden state of the encoder or
        decoder with a specified strategy.

        Args:
            encoded_inputs:
                A dictionary of encoded inputs.
            use_decoder_last_hidden_state:
                Whether to use the last hidden state of the decoder. If false,
                the last hidden state of the encoder will be used.
            strategy:
                The strategy to obtain an embedding for an object (e.g., column
                or row). The following strategies are supported:
                - "first": Use the first token.
                - "last": Use the last (non-pad) token.
                - "average": Average all (non-pad) tokens.

        Returns:
            embeddings:
                A tensor of shape (<number of embeddings>, <embedding size>).
        """

        encoded_inputs["decoder_input_ids"] = self.model._shift_right(
            encoded_inputs["input_ids"]
        )

        for key in encoded_inputs:
            encoded_inputs[key] = encoded_inputs[key].to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded_inputs)

        if use_decoder_last_hidden_state:
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs.encoder_last_hidden_state

        if strategy == "first":
            embeddings = last_hidden_state[:, 0]
        else:
            raise NotImplementedError

        assert embeddings.dim() == 2, (
            f"`embeddings` should be a 2D tensor but has {embeddings.dim()} "
            "dimensions."
        )
        assert embeddings.shape[1] == self.get_embedding_dimension(), (
            "The second dimension  of `embeddings` has an unexpected size of "
            f"{embeddings.shape[1]} while the expected size is "
            f"{self.model.get_embedding_dimension()}."
        )

        return embeddings


class T5ModelWrapper(T5FamilyModelWrapper):
    """Model wrapper for T5 models."""

    def get_model(self) -> T5Model:
        try:
            model = T5Model.from_pretrained(
                self.model_name, local_files_only=True
            )
        except OSError:
            model = T5Model.from_pretrained(self.model_name)

        model = model.to(self.device)
        model.eval()

        return model

    def get_max_input_size(self) -> int:
        """
        There is no hard limit of input size for T5 models due to the use of
        relative positional embeddings. However, the model was pre-trained on
        sequences of length 512 and the memory consumption increases
        quadratically with the sequence length. See discussions in
        https://github.com/huggingface/transformers/issues/5204
        """

        return 512

    def get_embedding_dimension(self) -> int:
        return self.model.config.d_model
