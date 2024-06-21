import pandas as pd

from observatory.preprocessing.preprocessing_wrapper import PreprocessingWrapper


def convert_table_to_col_list(table: pd.DataFrame) -> list[str]:
    """Convert a table to a list of columns where each column is a string.

    Args:
        table: A pandas DataFrame representing a table.

    Returns:
        A list of rows where each row is represented as a string
        consisting of column headers followed by column values.
    """

    cols = []

    for i in range(len(table.columns)):
        str_values = " ".join(table.iloc[:, i].astype(str).tolist())
        col_str = f"{table.columns[i]} {str_values}"
        cols.append(col_str)

    return cols


class ColumnwiseMaxRowsPreprocessor(PreprocessingWrapper):
    """Columnwise preprocessing for BERT-like models. The preprocessor attempts
    to serialize each table to a sequence of tokens (up to the maximum number
    of rows that fit within the model input size) as follows:

    [CLS]<col 1>[CLS]<col 2>[CLS]...[CLS][col n][SEP]
    """

    def is_fit(self, sample_table: pd.DataFrame) -> bool:
        """Check if a table fits within the maximum model input size.

        Args:
            sample_table: A pandas DataFrame representing a sample table.

        Returns:
            A boolean indicating if the table fits within the maximum model
            input size.
        """

        cols = convert_table_to_col_list(sample_table)
        current_tokens = []

        for col in cols:
            # Tokenize the column
            col_tokens = (
                [self.tokenizer.cls_token]
                + self.tokenizer.tokenize(col)
                + [self.tokenizer.sep_token]
            )

            # Check if adding new tokens would exceed the max input size
            if len(current_tokens) + len(col_tokens) > self.max_input_size:
                # If so, stop and return false
                return False
            else:
                # If not, remove the last [SEP] token and concatenate new tokens
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

    def truncate_columnwise(self, tables: list[pd.DataFrame]):
        """Truncate tables based on columnwise serialization to fit within the
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
                raise ValueError(
                    "The table is too wide and no single row can fit within "
                    "the maximum model input size. Consider splitting the "
                    "table columnwise."
                )

            truncated_tbl = tbl.iloc[:max_rows_fit, :]
            truncated_tables.append(truncated_tbl)

        return truncated_tables


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

    pass
