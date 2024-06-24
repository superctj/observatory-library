import pandas as pd
import torch

from transformers import PreTrainedTokenizer

from observatory.preprocessing.preprocessing_wrapper import PreprocessingWrapper


class RowwiseMaxRowsPreprocessor(PreprocessingWrapper):
    """Rowwise preprocessing for BERT-like models. The preprocessor attempts
    to serialize each table rowwise to a sequence of tokens (up to the maximum
    number of rows that fit within the model input size) as follows:

    [CLS]<row 1>[CLS]<row 2>[CLS]...[CLS][row n][SEP]
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
            A list of row texts following the template.
        """

        templated_rows = []

        for _, row in table.iterrows():
            if self.include_column_names:
                row_text = " ".join(
                    [f"{col} {val}" for col, val in zip(table.columns, row)]
                )
            else:
                row_text = " ".join([val for val in row])

            templated_rows.append(row_text)

        return templated_rows

    def is_fit(self, sample_table: pd.DataFrame) -> bool:
        """Check if a table fits within the maximum model input size.

        Args:
            sample_table: A pandas DataFrame representing a sample table.

        Returns:
            A boolean indicating if the table fits within the maximum model
            input size.
        """

        templated_rows = self.apply_text_template(sample_table)
        current_tokens = []

        for row in templated_rows:
            # Tokenize the row
            row_tokens = (
                [self.tokenizer.cls_token]
                + self.tokenizer.tokenize(row)
                + [self.tokenizer.sep_token]
            )

            # Check if adding new tokens would exceed the max input size
            if len(current_tokens) + len(row_tokens) > self.max_input_size:
                return False
            else:
                # Remove the last [SEP] token and concatenate new tokens
                current_tokens = current_tokens[:-1] + row_tokens

        return True

    def max_rows_fit(self, table: pd.DataFrame) -> int:
        """Compute the maximum number of rows that fit within the maximum model
        input size.

        Args:
            table: A pandas DataFrame representing a table.

        Returns:
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

    def truncate_rowwise(self, table: pd.DataFrame):
        """Truncate tables based on rowwise serialization to fit within the
        maximum model input size.

        Args:
            table: A table in Pandas data frame.

        Returns:
            truncated_tables: A truncated table.
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
