import pandas as pd
import torch

from transformers import PreTrainedTokenizer

from observatory.preprocessing.preprocessing_wrapper import PreprocessingWrapper


class ColumnwiseMaxRowsPreprocessor(PreprocessingWrapper):
    """Columnwise preprocessing for BERT-like models. The preprocessor attempts
    to serialize each table to a sequence of tokens (up to the maximum number
    of rows that fit within the model input size) as follows:

    [CLS]<col 1>[CLS]<col 2>[CLS]...[CLS][col n][SEP]
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_input_size: int,
        include_column_names: bool = True,
    ):
        super().__init__(tokenizer, max_input_size)

        self.include_column_names = include_column_names

    def apply_text_template(self, table: pd.DataFrame) -> list[str]:
        """Convert a table to a list of columns following a text template.

        Args:
            table: A Pandas DataFrame representing a table.

        Returns:
            A list of column texts following the template.
        """

        templated_cols = []

        for i in range(len(table.columns)):
            col_text = " ".join(table.iloc[:, i].tolist())

            if self.include_column_names:
                col_text = table.columns[i] + " " + col_text

            templated_cols.append(col_text)

        return templated_cols

    def is_fit(self, sample_table: pd.DataFrame) -> bool:
        """Check if a table fits within the maximum model input size.

        Args:
            sample_table: A pandas DataFrame representing a sample table.

        Returns:
            A boolean indicating if the table fits within the maximum model
            input size.
        """

        templated_cols = self.apply_text_template(sample_table)
        current_tokens = []

        for col in templated_cols:
            # Tokenize the column
            col_tokens = (
                [self.tokenizer.cls_token]
                + self.tokenizer.tokenize(col)
                + [self.tokenizer.sep_token]
            )

            # Check if adding new tokens would exceed the max input size
            if len(current_tokens) + len(col_tokens) > self.max_input_size:
                return False
            else:
                # Remove the last [SEP] token and append new tokens
                current_tokens = current_tokens[:-1] + col_tokens

        return True

    def max_rows_fit(self, table: pd.DataFrame) -> int:
        """Compute the maximum number of rows that fit within the maximum model
        input size.

        Args:
            table: A pandas DataFrame representing a table.

        Returns:
            The maximum number of rows that fit within the maximum model input
            size.
        """

        low = 0
        high = len(table)

        while low < high:
            mid = (low + high + 1) // 2  # middle point
            sample_table = table[:mid]  # sample table with `mid` rows

            if self.is_fit(sample_table):
                low = mid  # if it fits, try with more rows
            else:
                high = mid - 1  # if it doesn't fit, try with fewer rows

        # When low == high, we found the maximum number of rows
        return low

    def truncate_columnwise(self, table: pd.DataFrame):
        """Truncate a table based on columnwise serialization to fit within the
        maximum model input size.

        Args:
            table: A table in Pandas data frame.

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

    def serialize_columnwise(
        self, tables: list[pd.DataFrame]
    ) -> tuple[dict, list]:
        """Serialize a table columnwise to a sequence of tokens.

        Args:
            table: A table in Pandas data frame.

        Returns:
            encoded_inputs:
                A dictionary containing encoded inputs.
            cls_positions:
                Positions of the [CLS] tokens in the serialized sequence.
        """

        batch_input_ids = []
        batch_attention_masks = []
        batch_cls_positions = []

        for tbl in tables:
            truncated_tbl = self.truncate_columnwise(tbl)
            templated_cols = self.apply_text_template(truncated_tbl)

            input_tokens = []
            cls_positions = []

            for col in templated_cols:
                col_tokens = self.tokenizer.tokenize(col)
                col_tokens = (
                    [self.tokenizer.cls_token]
                    + col_tokens
                    + [self.tokenizer.sep_token]
                )

                new_length = len(input_tokens) + len(col_tokens)
                if new_length > self.max_input_size:
                    raise ValueError(
                        f"The length of the serialized table ({new_length}) "
                        "exceeds the maximum model input size "
                        f"({self.max_input_size}). Consider splitting the "
                        "table columnwise to fit the model input size."
                    )
                else:
                    input_tokens = input_tokens[:-1] + col_tokens
                    cls_positions.append(len(input_tokens) - len(col_tokens))

            # pad the sequence if necessary
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
            batch_cls_positions.append(cls_positions)

        encoded_inputs = {
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "attention_mask": torch.stack(batch_attention_masks, dim=0),
        }

        return encoded_inputs, batch_cls_positions


class ColumnwiseDocumentFrequencyBasedPreprocessor(PreprocessingWrapper):
    """Frequency-based columnwise preprocessing from Dong et al. DeepJoin
    (https://www.vldb.org/pvldb/vol16/p2458-dong.pdf).

    Each column is considered as a set of unique cells and serialized to a
    sequence of tokens following the template:

    <table name>. <column name> contains <n> values (<max number of
    characters in cell values, <min>, <average>): <cell 1>, <cell 2>, ...,
    <cell n>.

    If the length of the serialized sequence exceeds the model input size, more
    frequent cell values are preserved up to the maximum number of tokens
    allowed. Here, the frequency is defined as document frequency, i.e., the
    number of columns in the table corpus that have the cell value.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_input_size: int,
        cell_frequencies: dict[str, int],
        include_table_name: bool = True,
        include_column_names: bool = True,
        include_column_stats: bool = True,
    ):
        super().__init__(tokenizer, max_input_size)

        self.include_table_name = include_table_name
        self.include_column_names = include_column_names
        self.include_column_stats = include_column_stats
        self.cell_frequencies = cell_frequencies

    def is_fit(self):
        pass

    def get_sorted_values_per_column(self, table: pd.DataFrame) -> list[str]:
        sorted_values_per_column = []

        for i in range(len(table.columns)):
            col_values = table.iloc[:, i].unique()

            # sort column values by their frequencies
            col_values = sorted(
                col_values,
                key=lambda x: self.cell_frequencies.get(x, 0),
                reverse=True,
            )

            sorted_values_per_column.append(col_values)

        return sorted_values_per_column

    def apply_text_template(self, table: pd.DataFrame) -> str:
        """Convert a table to a list of columns following a text template.

        Args:
            table:
                A Pandas DataFrame representing a table.

        Returns:
            templated_cols:
                A list of column texts following the template.
        """

        table_name = table.attrs.get("name")

        if not table_name and self.include_table_name:
            raise ValueError(
                "Table name is missing while `include_table_name` is set to "
                "True` when initializing the preprocessor."
            )

        sorted_values_per_column = self.get_sorted_values_per_column(table)

        templated_cols = []

        for i, col_values in enumerate(sorted_values_per_column):
            col_text = ""

            if self.include_table_name:
                col_text += f"{table_name}. "

            if self.include_column_names:
                col_text += (
                    f"{table.columns[i]} contains {len(col_values)} values"
                )

                if self.include_column_stats:
                    col_text += (
                        f" ({len(min(col_values, key=len))}, "
                        f"{len(max(col_values, key=len))}, "
                        f"{sum(map(len, col_values)) / len(col_values)}): "
                    )
                else:
                    col_text += ": "

            col_text += ", ".join(col_values)
            templated_cols.append(col_text)

        return templated_cols

    def serialize_columnwise(self, tables: list[pd.DataFrame]) -> dict:
        """Batch serialize a list of tables columnwise to a sequence of tokens.

        Args:
            tables:
                A list of tables in Pandas data frame.

        Returns:
            encoded_inputs:
                A dictionary containing encoded inputs.
        """

        templated_cols = []

        for tbl in tables:
            templated_cols.extend(self.apply_text_template(tbl))

        encoded_inputs = self.tokenizer(
            templated_cols, padding=True, truncation=True, return_tensors="pt"
        )

        return encoded_inputs
