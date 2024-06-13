import pandas as pd
import torch

from transformers import BertModel, BertTokenizer

from observatory.models.model_wrapper import ModelWrapper
from observatory.preprocessing.columnwise import convert_table_to_col_list
from observatory.preprocessing.rowwise import convert_table_to_row_list


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

        cols = convert_table_to_col_list(table)
        input_tokens = []
        cls_positions = []

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

    def rowwise_serialization(
        self, table: pd.DataFrame
    ) -> tuple[list[str], list[int]]:
        """Serialize a table rowwise and inject special tokens.

        Args:
            table: A pandas DataFrame representing a table.

        Returns:
            input_tokens: A list of tokens representing the serialized table.
            cls_positions: A list of positions of [CLS] tokens.
        """

        rows = convert_table_to_row_list(table)
        input_tokens = []
        cls_positions = []

        for row in rows:
            row_tokens = self.tokenizer.tokenize(row)
            row_tokens = (
                [self.tokenizer.cls_token]
                + row_tokens
                + [self.tokenizer.sep_token]
            )

            if len(input_tokens) + len(row_tokens) > self.max_input_size:
                raise ValueError(
                    "The length of the serialized table exceeds the maximum input size. Please preprocess the table to fit the model input size."  # noqa: E501
                )
            else:
                input_tokens = input_tokens[:-1] + row_tokens
                cls_positions.append(len(input_tokens) - len(row_tokens))

        if len(input_tokens) < self.max_input_size:
            pad_length = self.max_input_size - len(input_tokens)
            input_tokens += [self.tokenizer.pad_token] * pad_length

        return input_tokens, cls_positions

    def infer_column_embeddings(
        self, tables: list[pd.DataFrame], batch_size: int
    ) -> list[list[torch.Tensor]]:
        """Column embedding inference.

        Args:
            tables: A list of tables.
            batch_size: The batch size for inference.

        Returns:
            all_embeddings: A list of lists of column embeddings where each
            inner list corresponds to a table.
        """

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

    def infer_row_embeddings(
        self, tables: list[pd.DataFrame], batch_size: int
    ) -> list[list[torch.Tensor]]:
        """Row embedding inference.

        Args:
            tables: A list of tables.
            batch_size: The batch size for inference.

        Returns:
            all_embeddings: A list of lists of row embeddings where each
            inner list corresponds to a table.
        """

        num_tables = len(tables)
        all_embeddings = []

        batch_input_ids = []
        batch_attention_masks = []
        batch_cls_positions = []

        for tbl_idx, tbl in enumerate(tables):
            input_tokens, cls_positions = self.rowwise_serialization(tbl)

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

                batch_last_hidden_states = outputs.last_hidden_state

                for i, last_hidden_state in enumerate(batch_last_hidden_states):
                    cls_embeddings = []

                    for pos in batch_cls_positions[i]:
                        cls_embeddings.append(
                            last_hidden_state[pos, :].detach().cpu()
                        )

                    all_embeddings.append(cls_embeddings)

        return all_embeddings

    def infer_table_embeddings(self):
        pass

    def infer_cell_embeddings(self):
        pass
