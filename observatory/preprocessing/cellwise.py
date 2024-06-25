import pandas as pd
import torch

from transformers import PreTrainedTokenizer

from observatory.preprocessing.preprocessing_wrapper import PreprocessingWrapper


class CellwiseMaxRowsPreprocessor(PreprocessingWrapper):
    """Preprocessing for cell embedding inference of BERT-like models. The
    preprocessor attempts to serialize each table columnwise or rowwise to a
    sequence of tokens (up to the maximum number of rows that fit within the
    model input size).

    Due to the large number of cells in a table, instead of inserting a [CLS]
    token for each cell, we keep track of cell tokens span and obtain cell
    embeddings by averaging the embeddings of the tokens in each cell.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_input_size: int,
        by_row: bool,
        include_table_name: bool = True,
        include_column_names: bool = True,
    ):
        super().__init__(tokenizer, max_input_size)

        self.by_row = by_row
        self.include_table_name = include_table_name
        self.include_column_names = include_column_names

    def apply_text_template(self, table: pd.DataFrame) -> list[list[str]]:
        """Convert a table to lists of cells rowwise or columnwise follwing a
        text template.

        Args:
            table:
                A Pandas DataFrame representing a table.

        Returns:
            templated_table:
                Lists of cell values.
        """

        templated_cells = []

        if self.by_row:
            for row in table.itertuples(index=False):
                if self.include_column_names:
                    row_cells = []

                    for col_name, val in zip(table.columns, row):
                        row_cells.append(col_name)
                        row_cells.append(val)
                else:
                    row_cells = [val for val in row]

                templated_cells.append(row_cells)
        else:
            for i in range(len(table.columns)):
                col_cells = table.iloc[:, i].tolist()

                if self.include_column_names:
                    col_cells = [table.columns[i]] + col_cells

                templated_cells.append(col_cells)

        return templated_cells

    def is_fit(
        self,
        table: pd.DataFrame,
    ) -> bool:
        """Check if a table fits within the maximum model input size.

        Args:
            sample_table: A pandas DataFrame representing a sample table.
            by_row:
                Whether to serialize the tables by row (if false, tables will
                be serialized by column).
            include_headers:
                Whether to include column headers in the serialized table.

        Returns:
            A boolean indicating if the table fits within the maximum model
            input size.
        """

        templated_cells = self.apply_text_template(table)

        # Start from 2 to account for `cls_token` and `sep_token`
        current_length = 2

        for cells in templated_cells:
            for c in cells:
                c_tokens = self.tokenizer.tokenize(c)

                # Check if adding new tokens would exceed the max input size
                if current_length + len(c_tokens) > self.max_input_size:
                    return False
                else:
                    current_length += len(c_tokens)

        return True

    def max_rows_fit(self, table: pd.DataFrame) -> bool:
        """Compute the maximum number of rows that fit within the model input
        size based on rowwise or columnwise serialization.

        Args:
            table:
                A pandas DataFrame representing a table.

        Returns:
            low:
                The maximum number of rows that fit within the model input size.
        """

        low = 0
        high = len(table)

        while low < high:
            mid = (low + high + 1) // 2  # middle point
            sample_table = table.iloc[:mid, :]  # sample table with `mid` rows

            if self.is_fit(sample_table):
                low = mid  # if it fits, try with more rows
            else:
                high = mid - 1  # if it doesn't fit, try with fewer rows

        # When low == high, we found the maximum number of rows
        return low

    def truncate(self, table: pd.DataFrame) -> pd.DataFrame:
        """Truncate a table based on rowwise or columnwise serialization to fit
        within the maximum model input size.

        Args:
            table:
                A Pandas DataFrame representing a table.

        Returns:
            truncated_table: A truncated table.
        """

        max_rows_fit = self.max_rows_fit(table)

        if max_rows_fit < 1:
            raise ValueError(
                "The table is too wide to fit within the maximum model input "
                "size. Consider splitting the table columnwise."
            )
        else:
            truncated_table = table.iloc[:max_rows_fit, :]

        return truncated_table

    def serialize(
        self, tables: list[pd.DataFrame]
    ) -> tuple[dict, list[list[tuple[int]]]]:
        """Serialize each table rowwise or columnwise to a sequence of tokens.

        Args:
            tables:
                A list of tables.

        Returns:
            encoded_inputs:
                A dictionary containing encoded inputs.
            batch_cell_positions:
                Lists of cell token spans in each serialized sequence. Each cell
                span is represented as a tuple of start (inclusive) and end
                (non-inclusive) indices.
        """

        batch_input_ids = []
        batch_attention_masks = []
        batch_cell_positions = []

        for tbl in tables:
            # Truncate tables to fit within the maximum model input size
            truncated_tbl = self.truncate(tbl)

            # Serialize the table by applying the text template
            templated_cells = self.apply_text_template(truncated_tbl)

            input_tokens = [self.tokenizer.cls_token]
            cell_positions = []

            if self.include_table_name:
                tbl_name = tbl.attrs.get("name")

                if not tbl_name:
                    raise ValueError(
                        "Table name is missing while `include_table_name` is "
                        "set to True when initializing the preprocessor."
                    )

                input_tokens += self.tokenizer.tokenize(tbl_name)

                if self.by_row:
                    if self.include_column_names:
                        for row_cells in templated_cells:
                            # The length should be even due to equal number of
                            # column names and cell values
                            assert len(row_cells) % 2 == 0

                            for i in range(0, len(row_cells), 2):
                                col_name = row_cells[i]
                                cell = row_cells[i + 1]

                                input_tokens += self.tokenizer.tokenize(
                                    col_name
                                )
                                start = len(input_tokens)
                                input_tokens += self.tokenizer.tokenize(cell)
                                cell_positions.append(
                                    (start, len(input_tokens))
                                )
                    else:
                        for row_cells in templated_cells:
                            for cell in row_cells:
                                start = len(input_tokens)
                                input_tokens += self.tokenizer.tokenize(cell)
                                cell_positions.append(
                                    (start, len(input_tokens))
                                )
                else:
                    for col_cells in templated_cells:
                        if self.include_column_names:
                            col_name = col_cells[0]
                            col_cells = col_cells[1:]

                            input_tokens += self.tokenizer.tokenize(col_name)

                        for cell in col_cells:
                            start = len(input_tokens)
                            input_tokens += self.tokenizer.tokenize(cell)
                            cell_positions.append((start, len(input_tokens)))

            input_tokens += [self.tokenizer.sep_token]

            if len(input_tokens) > self.max_input_size:
                raise ValueError(
                    f"The length of the serialized table ({len(input_tokens)}) "
                    f"exceeds the maximum input size ({self.max_input_size}). "
                    "Consider splitting the table columnwise to fit the model "
                    "input size."
                )

            if len(input_tokens) < self.max_input_size:
                pad_length = self.max_input_size - len(input_tokens)
                input_tokens += [self.tokenizer.pad_token] * pad_length

            input_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(input_tokens)
            )
            attention_mask = torch.tensor(
                [
                    1 if token != self.tokenizer.pad_token else 0
                    for token in input_tokens
                ]
            )

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            batch_cell_positions.append(cell_positions)

        encoded_inputs = {
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "attention_mask": torch.stack(batch_attention_masks, dim=0),
        }

        return encoded_inputs, batch_cell_positions
