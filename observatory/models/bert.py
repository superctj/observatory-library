import pandas as pd
import torch

from transformers import BertModel, BertTokenizer

from observatory.models.model_wrapper import ModelWrapper
from observatory.preprocessing.columnwise import convert_table_to_col_list


class BERTModelWrapper(ModelWrapper):
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

    def get_tokenizer(self) -> BertTokenizer:
        try:
            tokenizer = BertTokenizer.from_pretrained(
                self.model_name, local_files_only=True
            )
        except OSError:
            tokenizer = BertTokenizer.from_pretrained(self.model_name)

        return tokenizer

    def get_max_input_size(self) -> int:
        return 512

    def columnwise_serialization(
        self, table: pd.DataFrame
    ) -> tuple[list[str], list[int]]:
        """Serialize a table columnwise and inject special tokens.

        Args:
            table: A pandas DataFrame representing a table.

        Returns:
            input_tokens: A list of tokens representing the serialized table.
            cls_positions: A list of positions of [CLS] tokens.
        """

        input_tokens = []
        cls_positions = []

        cols = convert_table_to_col_list(table)
        for col in cols:
            col_tokens = self.tokenizer.tokenize(col)
            col_tokens = (
                [self.tokenizer.cls_token]
                + col_tokens
                + [self.tokenizer.sep_token]
            )

            if len(input_tokens) + len(col_tokens) > self.max_input_size:
                raise ValueError(
                    "The length of the serialized table exceeds the maximum input size. Please preprocess the table to fit the model input size."  # noqa: E501
                )
            else:
                input_tokens = input_tokens[:-1] + col_tokens
                cls_positions.append(len(input_tokens) - len(col_tokens))

        # pad the sequence if necessary
        if len(input_tokens) < self.max_input_size:
            pad_length = self.max_input_size - len(input_tokens)
            input_tokens += [self.tokenizer.pad_token] * pad_length

        return input_tokens, cls_positions

    def get_column_embeddings(
        self, tables: pd.DataFrame, batch_size: int
    ) -> list[list[torch.Tensor]]:
        """Column embedding inference."""

        num_tables = len(tables)
        all_embeddings = []

        batch_input_ids = []
        batch_attention_masks = []
        batch_cls_positions = []

        for tbl_idx, tbl in enumerate(tables):
            input_tokens, cls_positions = self.columnwise_serialization(tbl)

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            attention_mask = [
                1 if token != self.tokenizer.pad_token else 0
                for token in input_tokens
            ]

            batch_input_ids.append(torch.tensor(input_ids))
            batch_attention_masks.append(torch.tensor(attention_mask))
            batch_cls_positions.append(cls_positions)

            if len(batch_input_ids) == batch_size or tbl_idx + 1 == num_tables:
                batch_input_ids_tensor = torch.stack(batch_input_ids, dim=0).to(
                    self.device
                )
                batch_attention_masks_tensor = torch.stack(
                    batch_attention_masks, dim=0
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=batch_input_ids_tensor,
                        attention_mask=batch_attention_masks_tensor,
                    )

                batch_last_hidden_state = outputs.last_hidden_state

                for i, last_hidden_state in enumerate(batch_last_hidden_state):
                    cls_embeddings = []

                    for pos in batch_cls_positions[i]:
                        cls_embeddings.append(
                            last_hidden_state[pos, :].detach().cpu()
                        )

                    all_embeddings.append(cls_embeddings)

        return all_embeddings
