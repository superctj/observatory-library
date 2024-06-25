import os
import unittest

import torch

from torch.utils.data import DataLoader

from observatory.datasets.wikitables import WikiTablesDataset, collate_fn
from observatory.models.bert_family import BertModelWrapper
from observatory.preprocessing.columnwise import ColumnwiseMaxRowsPreprocessor
from observatory.preprocessing.rowwise import RowwiseMaxRowsPreprocessor
from observatory.preprocessing.tablewise import TablewiseMaxRowsPreprocessor
from observatory.utils.test_util import (
    assert_num_embeddings_matches_num_columns,
    assert_num_embeddings_matches_num_rows,
    assert_num_embeddings_matches_num_tables,
)


class TestBertEmbeddingInference(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "sample_data/wiki_tables",
        )
        metadata_filepath = os.path.join(data_dir, "table_inventory.csv")
        wikitables_dataset = WikiTablesDataset(data_dir, metadata_filepath)

        model_name = "google-bert/bert-base-uncased"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_wrapper = BertModelWrapper(model_name, device)

        self.wikitables_dataloader = DataLoader(
            wikitables_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # # Preprocessor for inferring cell embeddings
        # self.cell_max_rows_preprocessor = CellMaxRowsPreprocessor(
        #     tokenizer=self.model_wrapper.tokenizer,
        #     max_input_size=self.model_wrapper.max_input_size,
        # )

    def test_infer_column_embeddings(self):
        # Max rows-based preprocessor for inferring column embeddings
        columnwise_max_rows_preprocessor = ColumnwiseMaxRowsPreprocessor(
            tokenizer=self.model_wrapper.tokenizer,
            max_input_size=self.model_wrapper.max_input_size,
            include_column_names=True,
        )

        for batch_tables in self.wikitables_dataloader:
            encoded_inputs, cls_positions = (
                columnwise_max_rows_preprocessor.serialize_columnwise(
                    batch_tables
                )
            )

            column_embeddings = self.model_wrapper.infer_embeddings(
                encoded_inputs, cls_positions
            )

            assert_num_embeddings_matches_num_columns(
                batch_tables, column_embeddings
            )

    def test_infer_row_embeddings(self):
        # Preprocessor for inferring row embeddings
        rowwise_max_rows_preprocessor = RowwiseMaxRowsPreprocessor(
            tokenizer=self.model_wrapper.tokenizer,
            max_input_size=self.model_wrapper.max_input_size,
            include_column_names=True,
        )

        for batch_tables in self.wikitables_dataloader:
            encoded_inputs, cls_positions = (
                rowwise_max_rows_preprocessor.serialize_rowwise(batch_tables)
            )

            row_embeddings = self.model_wrapper.infer_embeddings(
                encoded_inputs, cls_positions
            )

            assert_num_embeddings_matches_num_rows(
                cls_positions, row_embeddings
            )

    def test_infer_table_embeddings(self):
        # Max rows-based preprocessor for inferring table embeddings
        tablewise_max_rows_preprocessor = TablewiseMaxRowsPreprocessor(
            tokenizer=self.model_wrapper.tokenizer,
            max_input_size=self.model_wrapper.max_input_size,
            by_row=True,
            include_table_name=True,
            include_column_names=True,
        )

        for batch_tables in self.wikitables_dataloader:
            encoded_inputs, cls_positions = (
                tablewise_max_rows_preprocessor.serialize(batch_tables)
            )

            # Here we do not need to pass in positions of [CLS] tokens as we
            # know there is only one [CLS] token at the beginning of each
            # sequence that represents the entire table
            table_embeddings = self.model_wrapper.infer_embeddings(
                encoded_inputs
            )

            assert_num_embeddings_matches_num_tables(
                batch_tables, table_embeddings
            )

            assert_num_embeddings_matches_num_tables(
                cls_positions, table_embeddings
            )

            # Passing in positions of [CLS] tokens should yield the same results
            same_table_embeddings = self.model_wrapper.infer_embeddings(
                encoded_inputs, cls_positions
            )

            assert torch.equal(table_embeddings, same_table_embeddings)

    # def test_infer_cell_embeddings(self):
    #     def assert_num_cells_matches_num_embeddings(
    #         tables: list[pd.DataFrame], cell_embeddings: list[torch.Tensor]
    #     ):
    #         """Assert the number of cells in each table matches the number of
    #         cell embeddings."""
    #         for i, tbl in enumerate(tables):
    #             num_cells = tbl.shape[0] * tbl.shape[1]
    #             assert len(cell_embeddings[i]) == num_cells

    #     # Include headers in serialized tables
    #     include_headers = True

    #     # Serialize tables by column to infer cell embeddings
    #     by_row = False
    #     truncated_tables = self.cell_max_rows_preprocessor.truncate(
    #         self.wikitables_dataset.all_tables, by_row, include_headers
    #     )
    #     cell_embeddings = self.model_wrapper.infer_cell_embeddings(
    #         truncated_tables,
    #         serialize_by_row=by_row,
    #         include_headers=include_headers,
    #         batch_size=32,
    #     )

    #     assert_num_cells_matches_num_embeddings(
    #         truncated_tables, cell_embeddings
    #     )

    #     # Serialize tables by row to infer cell embeddings
    #     by_row = True
    #     truncated_tables = self.cell_max_rows_preprocessor.truncate(
    #         self.wikitables_dataset.all_tables, by_row, include_headers
    #     )
    #     cell_embeddings = self.model_wrapper.infer_cell_embeddings(
    #         truncated_tables,
    #         serialize_by_row=by_row,
    #         include_headers=include_headers,
    #         batch_size=32,
    #     )

    #     assert_num_cells_matches_num_embeddings(
    #         truncated_tables, cell_embeddings
    #     )

    #     # Do not include headers in serialized tables
    #     include_headers = False

    #     # Serialize tables by column to infer cell embeddings
    #     by_row = False
    #     truncated_tables = self.cell_max_rows_preprocessor.truncate(
    #         self.wikitables_dataset.all_tables, by_row, include_headers
    #     )
    #     cell_embeddings = self.model_wrapper.infer_cell_embeddings(
    #         truncated_tables,
    #         serialize_by_row=by_row,
    #         include_headers=include_headers,
    #         batch_size=32,
    #     )

    #     assert_num_cells_matches_num_embeddings(
    #         truncated_tables, cell_embeddings
    #     )

    #     # Serialize tables by row to infer cell embeddings
    #     by_row = True
    #     truncated_tables = self.cell_max_rows_preprocessor.truncate(
    #         self.wikitables_dataset.all_tables, by_row, include_headers
    #     )
    #     cell_embeddings = self.model_wrapper.infer_cell_embeddings(
    #         truncated_tables,
    #         serialize_by_row=by_row,
    #         include_headers=include_headers,
    #         batch_size=32,
    #     )

    #     assert_num_cells_matches_num_embeddings(
    #         truncated_tables, cell_embeddings
    #     )
