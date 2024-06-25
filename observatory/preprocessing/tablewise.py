import pandas as pd

from transformers import PreTrainedTokenizer

from observatory.preprocessing.preprocessing_wrapper import PreprocessingWrapper


class TablewiseMaxRowsPreprocessor(PreprocessingWrapper):
    """Preprocessing for table embedding inference of BERT-like models. The
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

    def apply_text_template(self, table: pd.DataFrame) -> str:
        """Convert a table to a text sequence following a template.

        Args:
            table:
                A Pandas DataFrame representing a table.

        Returns:
            templated_table:
                A text sequence representing the flattened table.
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

    def is_fit(self, table: pd.DataFrame) -> bool:
        """Check if a table fits within the maximum model input size based on
        rowwise or columnwise serialization.

        Args:
            table: A Pandas DataFrame representing a table.

        Returns:
            A boolean indicating if the table fits within the maximum model
            input size.
        """

        templated_table = self.apply_text_template(table)
        table_tokens = (
            [self.tokenizer.cls_token]
            + self.tokenizer.tokenize(templated_table)
            + [self.tokenizer.sep_token]
        )

        if len(table_tokens) > self.max_input_size:
            return False
        else:
            return True

    def max_rows_fit(self, table: pd.DataFrame) -> int:
        """Compute the maximum number of rows that fit within the model input
        size based on rowwise or columnwise serialization.

        Args:
            table: A Pandas DataFrame representing a table.

        Returns:
            low:
                The maximum number of rows that fit within the model input size.
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
    ) -> tuple[dict, list[list[int]]]:
        """Serialize each table rowwise or columnwise to a sequence of tokens.

        Args:
            tables:
                A list of tables.

        Returns:
            encoded_inputs:
                A dictionary containing encoded inputs.
            batch_cls_positions:
                Lists of positions of [CLS] tokens in each serialized sequence.
        """

        templated_tables = [
            self.apply_text_template(self.truncate(tbl)) for tbl in tables
        ]

        encoded_inputs = self.tokenizer(
            templated_tables,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        batch_cls_positions = [[0]] * len(templated_tables)

        return encoded_inputs, batch_cls_positions
