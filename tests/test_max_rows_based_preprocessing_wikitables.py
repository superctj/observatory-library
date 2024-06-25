import os
import unittest

import torch

from torch.utils.data import DataLoader

from observatory.datasets.wikitables import WikiTablesDataset, collate_fn
from observatory.models.bert_family import BertModelWrapper
from observatory.preprocessing.columnwise import ColumnwiseMaxRowsPreprocessor
from observatory.preprocessing.rowwise import RowwiseMaxRowsPreprocessor
from observatory.utils.test_util import (
    assert_num_embeddings_matches_number_columns,
    assert_num_embeddings_matches_number_rows,
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

        # Max rows-based preprocessor for inferring column embeddings
        self.columnwise_max_rows_preprocessor = ColumnwiseMaxRowsPreprocessor(
            tokenizer=self.model_wrapper.tokenizer,
            max_input_size=self.model_wrapper.max_input_size,
            include_column_names=True,
        )

        # Preprocessor for inferring row embeddings
        self.rowwise_max_rows_preprocessor = RowwiseMaxRowsPreprocessor(
            tokenizer=self.model_wrapper.tokenizer,
            max_input_size=self.model_wrapper.max_input_size,
            include_column_names=True,
        )

        # # Preprocessor for inferring table embeddings
        # self.table_max_rows_preprocessor = TableMaxRowsPreprocessor(
        #     tokenizer=self.model_wrapper.tokenizer,
        #     max_input_size=self.model_wrapper.max_input_size,
        # )

        # # Preprocessor for inferring cell embeddings
        # self.cell_max_rows_preprocessor = CellMaxRowsPreprocessor(
        #     tokenizer=self.model_wrapper.tokenizer,
        #     max_input_size=self.model_wrapper.max_input_size,
        # )

    def test_infer_column_embeddings(self):
        for batch_tables in self.wikitables_dataloader:
            encoded_inputs, cls_positions = (
                self.columnwise_max_rows_preprocessor.serialize_columnwise(
                    batch_tables
                )
            )

            column_embeddings = self.model_wrapper.infer_embeddings(
                encoded_inputs, cls_positions
            )

            assert_num_embeddings_matches_number_columns(
                batch_tables, column_embeddings
            )

    def test_infer_row_embeddings(self):
        for batch_tables in self.wikitables_dataloader:
            encoded_inputs, cls_positions = (
                self.rowwise_max_rows_preprocessor.serialize_rowwise(
                    batch_tables
                )
            )

            row_embeddings = self.model_wrapper.infer_embeddings(
                encoded_inputs, cls_positions
            )

            assert_num_embeddings_matches_number_rows(
                cls_positions, row_embeddings
            )

    # def test_infer_table_embeddings(self):
    #     # Serialize tables by column to infer table embeddings
    #     truncated_tables = self.table_max_rows_preprocessor.truncate_columnwise(
    #         self.wikitables_dataset.all_tables
    #     )
    #     table_embeddings = self.model_wrapper.infer_table_embeddings(
    #         truncated_tables, serialize_by_row=False, batch_size=32
    #     )
    #     assert len(table_embeddings) == len(self.wikitables_dataset.all_tables)

    #     # Serialize tables by row to infer table embeddings
    #     truncated_tables = self.table_max_rows_preprocessor.truncate_rowwise(
    #         self.wikitables_dataset.all_tables
    #     )
    #     table_embeddings = self.model_wrapper.infer_table_embeddings(
    #         truncated_tables, serialize_by_row=True, batch_size=32
    #     )
    #     assert len(table_embeddings) == len(self.wikitables_dataset.all_tables)

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
