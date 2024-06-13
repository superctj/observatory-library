import pandas as pd

from observatory.preprocessing.preprocessing_wrapper import PreprocessingWrapper


def convert_table_to_row_list(table: pd.DataFrame) -> list[str]:
    """Convert a table to a list of columns where each column is a string.

    Args:
        table: A pandas DataFrame representing a table.

    Returns:
        A list of columns where each column is represented as a string
        consisting of the column header followed by column values.
    """

    rows = []

    for _, row in table.iterrows():
        row_str = " ".join(
            [f"{col} {str(val)}" for col, val in zip(table.columns, row)]
        )
        rows.append(row_str)

    return rows


class RowwiseMaxRowsPreprocessor(PreprocessingWrapper):
    """Rowwise preprocessing for BERT-like models. Each table is serialized
    to a string as follows:

    [CLS]<row 1>[CLS]<row 2>[CLS]...[CLS][row n][SEP]
    """

    def is_fit(self, sample_table: pd.DataFrame) -> bool:
        """Check if a table fits within the maximum model input size.

        Args:
            sample_table: A pandas DataFrame representing a sample table.

        Returns:
            A boolean indicating if the table fits within the maximum model
            input size.
        """

        rows = convert_table_to_row_list(sample_table)
        current_tokens = []

        for row in rows:
            # Tokenize row without special tokens
            row_tokens = self.tokenizer.tokenize(row)

            # [CLS] and [SEP] here are only placeholders (different models can
            # use different special tokens)
            row_tokens = ["[CLS]"] + row_tokens + ["[SEP]"]

            # Check if adding new tokens would exceed the max input size
            if len(current_tokens) + len(row_tokens) > self.max_input_size:
                # If so, stop and return false
                return False
            else:
                # If not, remove the last [SEP] token and concatenate new tokens
                current_tokens = current_tokens[:-1] + row_tokens

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

    def rowwise_truncation(self, tables: list[pd.DataFrame]):
        """Truncate tables based on rowwise serialization to fit within the
        maximum model input size.

        Args:
            tables: A list of tables.

        Returns:
            truncated_tables: A list of truncated tables.
        """

        truncated_tables = []

        for tbl in tables:
            max_rows_fit = self.max_rows_fit(tbl)

            if max_rows_fit < 1:
                # TODO: raise error of wide tables
                continue

            truncated_tbl = tbl.iloc[:max_rows_fit, :]
            truncated_tables.append(truncated_tbl)

        return truncated_tables
