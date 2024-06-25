import pandas as pd
import torch

from transformers import PreTrainedTokenizer

from observatory.preprocessing.preprocessing_wrapper import PreprocessingWrapper


def convert_table_to_str_columnwise(table: pd.DataFrame) -> str:
    """Convert a table to a string with columnwise serialization.

    Args:
        table: A pandas DataFrame representing a table.

    Returns:
        A string representing a columnwise serialized table.
    """

    cols = convert_table_to_col_list(table)

    return " ".join(cols)


def convert_table_to_str_rowwise(table: pd.DataFrame) -> str:
    """Convert a table to a string with rowwise serialization.

    Args:
        table: A pandas DataFrame representing a table.

    Returns:
        A string representing a rowwise serialized table.
    """

    rows = convert_table_to_row_list(table)

    return " ".join(rows)


class TablewiseMaxRowsPreprocessor(PreprocessingWrapper):
    """Preprocessing for table embedding inference of BERT-like model. The
    preprocessor attempts to serialize each table columnwise or rowwise to a
    sequence of tokens (up to the maximum number of rows that fit within the
    model input size).
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

    def apply_text_template(self, table: pd.DataFrame) -> list[str]:
        """Convert a table to a list of columns or rows following a text
        template

        Args:
            table:
                A Pandas DataFrame representing a table.

        Returns:
            templated_table:
                A list of column or row texts following the template.
        """

        if self.include_table_name:
            templated_table = table.attrs.get("name")
        else:
            templated_table = ""

        if self.by_row:
            for row in table.itertuples(index=False):
                if self.include_column_names:
                    row_text = " ".join(
                        [f"{col} {val}" for col, val in zip(table.columns, row)]
                    )
                else:
                    row_text = " ".join([val for val in row])

                if templated_table == "":
                    templated_table = row_text
                else:
                    templated_table += f" {row_text}"
        else:
            for i in range(len(table.columns)):
                col_text = " ".join(table.iloc[:, i].tolist())

                if self.include_column_names:
                    col_text = table.columns[i] + " " + col_text

                if templated_table == "":
                    templated_table = col_text
                else:
                    templated_table += f" {col_text}"

        return templated_table

    def is_fit_columnwise(self, sample_table: pd.DataFrame) -> bool:
        """Check if a table fits within the maximum model input size based on
        columnwise serialization.

        Args:
            sample_table: A pandas DataFrame representing a sample table.

        Returns:
            A boolean indicating if the table fits within the maximum model
            input size.
        """

        table_str = convert_table_to_str_columnwise(sample_table)
        table_tokens = (
            ["[CLS]"] + self.tokenizer.tokenize(table_str) + ["[SEP]"]
        )

        if len(table_tokens) > self.max_input_size:
            return False
        else:
            return True

    def is_fit_rowwise(self, sample_table: pd.DataFrame) -> bool:
        """Check if a table fits within the maximum model input size based on
        rowwise serialization.

        Args:
            sample_table: A pandas DataFrame representing a sample table.

        Returns:
            A boolean indicating if the table fits within the maximum model
            input size.
        """

        table_str = convert_table_to_str_rowwise(sample_table)
        table_tokens = (
            ["[CLS]"] + self.tokenizer.tokenize(table_str) + ["[SEP]"]
        )

        if len(table_tokens) > self.max_input_size:
            return False
        else:
            return True

    def max_rows_fit_columnwise(self, table: pd.DataFrame) -> int:
        """Compute the maximum number of rows that fit within the maximum model
        input size based on columnwise serialization.

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

            if self.is_fit_columnwise(sample_table):
                low = mid  # if it fits, try with more rows
            else:
                high = mid - 1  # if it doesn't fit, try with fewer rows

        # When low == high, we found the maximum number of rows
        return low

    def max_rows_fit_rowwise(self, table: pd.DataFrame) -> int:
        """Compute the maximum number of rows that fit within the maximum model
        input size based on rowwise serialization.

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

            if self.is_fit_rowwise(sample_table):
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
            max_rows_fit = self.max_rows_fit_columnwise(tbl)

            if max_rows_fit < 1:
                # TODO: raise error of wide tables
                continue

            truncated_tbl = tbl.iloc[:max_rows_fit, :]
            truncated_tables.append(truncated_tbl)

        return truncated_tables

    def truncate_rowwise(self, tables: list[pd.DataFrame]):
        """Truncate tables based on rowwise serialization to fit within the
        maximum model input size.

        Args:
            tables: A list of tables.

        Returns:
            truncated_tables: A list of truncated tables.
        """

        truncated_tables = []

        for tbl in tables:
            max_rows_fit = self.max_rows_fit_rowwise(tbl)

            if max_rows_fit < 1:
                # TODO: raise error of wide tables
                continue

            truncated_tbl = tbl.iloc[:max_rows_fit, :]
            truncated_tables.append(truncated_tbl)

        return truncated_tables

    def truncate(self, table: pd.DataFrame):
        pass
