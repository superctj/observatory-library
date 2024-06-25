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
            table:
                A Pandas DataFrame representing a table.

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

    def serialize_rowwise(
        self, tables: list[pd.DataFrame]
    ) -> tuple[dict, list[list[int]]]:
        """Serialize each table rowwise to a sequence of tokens.

        Args:
            tables:
                A list of tables.

        Returns:
            encoded_inputs:
                A dictionary containing encoded inputs.
            batch_cls_positions:
                Lists of positions of [CLS] tokens in each serialized sequence.
        """

        batch_input_ids = []
        batch_attention_masks = []
        batch_cls_positions = []

        for tbl in tables:
            # Truncate tables to fit within the maximum model input size
            truncated_tbl = self.truncate_rowwise(tbl)
            # Apply the text template
            templated_rows = self.apply_text_template(truncated_tbl)

            # Serialize each row to a sequence of tokens
            input_tokens = []
            cls_positions = []

            for row in templated_rows:
                row_tokens = (
                    [self.tokenizer.cls_token]
                    + self.tokenizer.tokenize(row)
                    + [self.tokenizer.sep_token]
                )

                new_length = len(input_tokens) + len(row_tokens)

                if new_length > self.max_input_size:
                    raise ValueError(
                        f"The length of the serialized table ({new_length}) "
                        "exceeds the maximum model input size "
                        f"({self.max_input_size}). Consider splitting the "
                        "table columnwise to fit the model input size."
                    )
                else:
                    input_tokens = input_tokens[:-1] + row_tokens
                    cls_positions.append(len(input_tokens) - len(row_tokens))

            # Pad the sequence if necessary
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
