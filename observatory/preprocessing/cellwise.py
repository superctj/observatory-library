import pandas as pd

from observatory.preprocessing.preprocessing_wrapper import PreprocessingWrapper


def convert_table_to_cell_lists_columnwise(
    table: pd.DataFrame, include_headers: bool
) -> list[list[str]]:
    """Convert a table to a list of lists of cell values where each inner list
    corresponds to a column.

    Args:
        table:
            A pandas DataFrame representing a table.
        include_headers:
            Whether to include column headers in the serialized table. If true,
            the first element of each inner list is the column header.

    Returns:
        A list of lists of cell values.
    """

    cells = []

    for i in range(len(table.columns)):
        if include_headers:
            col_list = [table.columns[i]] + table.iloc[:, i].tolist()
        else:
            col_list = table.iloc[:, i].tolist()

        cells.append(col_list)

    return cells


def convert_table_to_cell_lists_rowwise(
    table: pd.DataFrame, include_headers: bool
) -> list[list[str]]:
    """Convert a table to a list of lists of cell values where each inner list
    corresponds to a row.

    Args:
        table:
            A pandas DataFrame representing a table.
        include_headers:
            Whether to include column headers in the serialized table. If true,
            the first element of each inner list is the column header.

    Returns:
        A list of lists of cell values.
    """

    cells = []

    for _, row in table.iterrows():
        if include_headers:
            row_list = []

            for col_name, val in zip(table.columns, row):
                row_list.append(col_name)
                row_list.append(val)
        else:
            row_list = [val for val in row]

        cells.append(row_list)

    return cells


class CellMaxRowsPreprocessor(PreprocessingWrapper):
    """Preprocessing for cell embedding inference."""

    def is_fit(self, sample_table: pd.DataFrame) -> bool:
        pass

    def is_fit_columnwise(
        self, sample_table: pd.DataFrame, include_headers: bool
    ) -> bool:
        """Check if a table fits within the maximum model input size based on
        columnwise serialization.

        Args:
            sample_table: A pandas DataFrame representing a sample table.
            include_headers:
                Whether to include column headers in the serialized table.

        Returns:
            A boolean indicating if the table fits within the maximum model
            input size.
        """

        cell_lists = convert_table_to_cell_lists_columnwise(
            sample_table, include_headers
        )
        # Start from 2 to account for cls_token and sep_token
        current_len = 2

        for cells in cell_lists:
            col_str = " ".join(cells)
            col_tokens = self.tokenizer.tokenize(col_str)

            # Check if adding new tokens would exceed the max input size
            if current_len + len(col_tokens) > self.max_input_size:
                # If so, stop and return false
                return False
            else:
                current_len += len(col_tokens)

        return True

    def is_fit_rowwise(
        self, sample_table: pd.DataFrame, include_headers: bool
    ) -> bool:
        """Check if a table fits within the maximum model input size based on
        rowwise serialization.

        Args:
            sample_table: A pandas DataFrame representing a sample table.
            include_headers:
                Whether to include column headers in the serialized table.

        Returns:
            A boolean indicating if the table fits within the maximum model
            input size.
        """

        cell_lists = convert_table_to_cell_lists_rowwise(
            sample_table, include_headers
        )
        # Start from 2 to account for [CLS] and [SEP] tokens
        current_len = 2

        for cells in cell_lists:
            row_str = " ".join(cells)
            col_tokens = self.tokenizer.tokenize(row_str)

            # Check if adding new tokens would exceed the max input size
            if current_len + len(col_tokens) > self.max_input_size:
                # If so, stop and return false
                return False
            else:
                current_len += len(col_tokens)

        return True

    def max_rows_fit(
        self, table: pd.DataFrame, by_row: bool, include_headers: bool
    ) -> bool:
        """Compute the maximum number of rows that fit within the maximum model
        input size based on columnwise serialization.

        Args:
            table:
                A pandas DataFrame representing a table.
            by_row:
                Whether to serialize the tables by row (if false, tables will
                be serialized by column).
            include_headers:
                Whether to include table headers in the serialized table.

        Returns:
            low:
                The maximum number of rows that fit within the maximum model
                input size.
        """

        low = 0
        high = len(table)

        while low < high:
            mid = (low + high + 1) // 2  # middle point
            sample_table = table.iloc[:mid, :]  # sample table with `mid` rows

            if by_row:
                is_fit = self.is_fit_rowwise(sample_table, include_headers)
            else:
                is_fit = self.is_fit_columnwise(sample_table, include_headers)

            if is_fit:
                low = mid  # if it fits, try with more rows
            else:
                high = mid - 1  # if it doesn't fit, try with fewer rows

        # When low == high, we found the maximum number of rows
        return low

    def truncate(
        self,
        tables: list[pd.DataFrame],
        by_row: bool,
        include_headers: bool,
    ) -> list[pd.DataFrame]:
        """Truncate tables based on columnwise serialization to fit within the
        maximum model input size.

        Args:
            tables: A list of tables.
            by_row:
                Whether to serialize the tables by row (if false, tables will
                be serialized by column).
            include_headers:
                Whether to include column headers in the serialized table.

        Returns:
            truncated_tables: A list of truncated tables.
        """

        truncated_tables = []

        for tbl in tables:
            max_rows_fit = self.max_rows_fit(tbl, by_row, include_headers)

            if max_rows_fit < 1:
                raise ValueError(
                    "Table is too wide to fit within the maximum model input "
                    "size. Split the table columnwise into smaller tables."
                )

            truncated_tbl = tbl.iloc[:max_rows_fit, :]
            truncated_tables.append(truncated_tbl)

        return truncated_tables
