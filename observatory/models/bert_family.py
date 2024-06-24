import pandas as pd
import torch

from transformers import (
    AlbertModel,
    AutoTokenizer,
    BertModel,
    RobertaModel,
)

from observatory.models.model_wrapper import ModelWrapper
from observatory.preprocessing.cellwise import (
    convert_table_to_cell_lists_columnwise,
    convert_table_to_cell_lists_rowwise,
)
from observatory.preprocessing.rowwise import convert_table_to_row_list
from observatory.preprocessing.tablewise import (
    convert_table_to_str_columnwise,
    convert_table_to_str_rowwise,
)


class BERTFamilyModelWrapper(ModelWrapper):
    """Model wrapper for any BERT-like model whose tokenizer has valid
    attributes `cls_token`, `sep_token`, and `pad_token`.

    To use this class, inherit from it and implement the `get_model` method.
    Override the `get_max_input_size` method if the model input size cannot be
    obtained from the model config.
    """

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

    def serialize_rowwise(
        self, table: pd.DataFrame
    ) -> tuple[list[str], list[int]]:
        """Serialize a table rowwise and inject special tokens for inferring
        row embeddings.

        Args:
            table:
                A pandas DataFrame representing a table.

        Returns:
            input_tokens:
                A list of tokens representing the serialized table.
            cls_positions:
                A list of positions of [CLS] tokens.
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
                    "The length of the serialized table exceeds the model maximum input size. Preprocess the table to fit the model input size."  # noqa: E501
                )
            else:
                input_tokens = input_tokens[:-1] + row_tokens
                cls_positions.append(len(input_tokens) - len(row_tokens))

        if len(input_tokens) < self.max_input_size:
            pad_length = self.max_input_size - len(input_tokens)
            input_tokens += [self.tokenizer.pad_token] * pad_length

        return input_tokens, cls_positions

    def serialize_table(
        self, table: pd.DataFrame, by_row: bool = True
    ) -> tuple[list[str], list[int]]:
        """Serialize a table for inferring the table embedding.

        Args:
            table:
                A pandas DataFrame representing a table.
            by_row:
                Whether to serialize by row or by column.

        Returns:
            input_tokens:
                A list of tokens representing the serialized table.
            cls_positions:
                A list of positions of [CLS] tokens.
        """

        if by_row:
            table_str = convert_table_to_str_rowwise(table)
        else:
            table_str = convert_table_to_str_columnwise(table)

        input_tokens = (
            [self.tokenizer.cls_token]
            + self.tokenizer.tokenize(table_str)
            + [self.tokenizer.sep_token]
        )

        if len(input_tokens) > self.max_input_size:
            raise ValueError(
                "The length of the serialized table exceeds the maximum input size. Preprocess the table to fit the model input size."  # noqa: E501
            )
        else:
            pad_length = self.max_input_size - len(input_tokens)
            input_tokens += [self.tokenizer.pad_token] * pad_length

        return input_tokens, [0]

    def serialize_cellwise(
        self,
        table: pd.DataFrame,
        by_row: bool = True,
        include_headers: bool = True,
    ) -> tuple[list[str], list[tuple[int]]]:
        """Serialize a table for inferring the cell embeddings.

        Args:
            table:
                A pandas DataFrame representing a table.
            by_row:
                Whether to serialize by row or by column.
            include_headers:
                Whether to include column headers in the serialized table.

        Returns:
            input_tokens:
                A list of tokens representing the serialized table.
            cell_positions:
                A list of tuples representing the start (inclusive) and end
                (non-inclusive) positions of cell tokens.
        """

        input_tokens = [self.tokenizer.cls_token]
        cell_positions = []

        if by_row:
            cell_lists = convert_table_to_cell_lists_rowwise(
                table, include_headers
            )

            if include_headers:
                for row_cells in cell_lists:
                    # The number of cells should match the number of headers
                    assert len(row_cells) % 2 == 0

                    for i in range(0, len(row_cells), 2):
                        header = row_cells[i]
                        cell = row_cells[i + 1]

                        input_tokens += self.tokenizer.tokenize(header)
                        start = len(input_tokens)
                        input_tokens += self.tokenizer.tokenize(cell)
                        cell_positions.append((start, len(input_tokens)))
            else:
                for row_cells in cell_lists:
                    for cell in row_cells:
                        start = len(input_tokens)
                        input_tokens += self.tokenizer.tokenize(cell)
                        cell_positions.append((start, len(input_tokens)))
        else:
            cell_lists = convert_table_to_cell_lists_columnwise(
                table, include_headers
            )

            for col_cells in cell_lists:
                if include_headers:
                    col_header = col_cells[0]
                    col_cells = col_cells[1:]

                    input_tokens += self.tokenizer.tokenize(col_header)

                for cell in col_cells:
                    start = len(input_tokens)
                    input_tokens += self.tokenizer.tokenize(cell)
                    cell_positions.append((start, len(input_tokens)))

        input_tokens += [self.tokenizer.sep_token]

        if len(input_tokens) > self.max_input_size:
            raise ValueError(
                f"The length of the serialized table ({len(input_tokens)}) "
                f"exceeds the maximum input size ({self.max_input_size}). "
                "Preprocess the table to fit the model input size."
            )

        if len(input_tokens) < self.max_input_size:
            pad_length = self.max_input_size - len(input_tokens)
            input_tokens += [self.tokenizer.pad_token] * pad_length

        return input_tokens, cell_positions

    def infer_row_embeddings(
        self, tables: list[pd.DataFrame], batch_size: int
    ) -> list[list[torch.Tensor]]:
        """Row embedding inference.

        Args:
            tables:
                A list of tables.
            batch_size:
                The batch size for inference.

        Returns:
            all_embeddings:
                A list of lists of row embeddings where each inner list
                corresponds to a table.
        """

        num_tables = len(tables)
        all_embeddings = []

        batch_input_ids = []
        batch_attention_masks = []
        batch_cls_positions = []

        for tbl_idx, tbl in enumerate(tables):
            input_tokens, cls_positions = self.serialize_rowwise(tbl)

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
                        cls_embeddings.append(last_hidden_state[pos, :])

                    all_embeddings.append(cls_embeddings)

        return all_embeddings

    def infer_table_embeddings(
        self,
        tables: list[pd.DataFrame],
        serialize_by_row: bool,
        batch_size: int,
    ) -> list[torch.Tensor]:
        """Table embedding inference.

        Args:
            tables:
                A list of tables.
            serialize_by_row:
                Whether to serialize the tables by row (if false, tables will
                be serialized by column).
            batch_size:
                The batch size for inference.

        Returns:
            all_embeddings:
                A list of table embeddings.
        """

        num_tables = len(tables)
        all_embeddings = []

        batch_input_ids = []
        batch_attention_masks = []
        batch_cls_positions = []

        for tbl_idx, tbl in enumerate(tables):
            input_tokens, cls_positions = self.serialize_table(
                tbl, serialize_by_row
            )

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

                for last_hidden_state in batch_last_hidden_state:
                    all_embeddings.append(last_hidden_state[0, :])

        # The number of embeddings should match the number of tables
        assert len(all_embeddings) == num_tables

        return all_embeddings

    def infer_cell_embeddings(
        self,
        tables: list[pd.DataFrame],
        serialize_by_row: bool,
        include_headers: bool,
        batch_size: int,
    ):
        """Cell embedding inference.

        Args:
            tables:
                A list of tables.
            serialize_by_row:
                Whether to serialize the tables by row (if false, tables will
                be serialized by column).
            include_headers:
                Whether to include column headers in the serialized table.
            batch_size:
                The batch size for inference.

        Returns:
            all_embeddings:
                A list of lists of cell embeddings where each inner list
                corresponds to a table.

        """

        num_tables = len(tables)
        all_embeddings = []

        batch_input_ids = []
        batch_attention_masks = []
        batch_cell_positions = []

        for tbl_idx, tbl in enumerate(tables):
            input_tokens, cell_positions = self.serialize_cellwise(
                tbl, by_row=serialize_by_row, include_headers=include_headers
            )

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            attention_mask = [
                1 if token != self.tokenizer.pad_token else 0
                for token in input_tokens
            ]

            batch_input_ids.append(torch.tensor(input_ids))
            batch_attention_masks.append(torch.tensor(attention_mask))
            batch_cell_positions.append(cell_positions)

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
                    cell_embeddings = []

                    for pos in batch_cell_positions[i]:
                        cell_embeddings.append(
                            torch.mean(
                                last_hidden_state[pos[0] : pos[1], :], dim=0
                            )
                        )

                    all_embeddings.append(cell_embeddings)

        return all_embeddings

    def infer_embeddings(
        self,
        encoded_inputs: dict,
        cls_positions: list[list[int]] = None,
    ) -> torch.FloatTensor:

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

            embeddings = torch.stack(embeddings, dim=0)

        assert embeddings.dim() == 2, (
            "`embeddings` should be a 2D tensor but has "
            f"{cls_embeddings.dim()} dimensions."
        )
        assert embeddings.shape[0] == encoded_inputs["input_ids"].shape[0]
        assert embeddings.shape[1] == self.model.config.hidden_size

        return embeddings

    def batch_infer_embeddings(
        self,
        encoded_inputs: dict,
        batch_size: int,
        cls_positions: list[list[int]] = None,
    ) -> torch.FloatTensor:
        """Infer embeddings in batches.

        Args:
            encoded_inputs:
                A dictionary of encoded inputs.
            batch_size:
                The batch size for inference.
            cls_positions:
                Positions of [CLS] tokens per table.

        Returns:
            embeddings:
                A tensor of embeddings or a list of tensors of embeddings.
        """

        num_inputs = encoded_inputs["input_ids"].shape[0]
        embeddings = torch.zeros(
            (num_inputs, self.model.config.hidden_size), dtype=torch.float
        )

        num_batches = (
            num_inputs // batch_size
            if num_inputs % batch_size == 0
            else num_inputs // batch_size + 1
        )

        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size

            batch_encoded_inputs = {
                key: value[start:end] for key, value in encoded_inputs.items()
            }

            if cls_positions:
                batch_cls_positions = cls_positions[start:end]
            else:
                batch_cls_positions = None

            batch_embeddings = self.infer_embeddings(
                batch_encoded_inputs, batch_cls_positions
            )

            embeddings[start:end] = batch_embeddings

        return embeddings


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
