import os
import unittest

import pandas as pd
import torch

from observatory.datasets.wikitables import WikiTables
from observatory.models.bert_family import RobertaModelWrapper
from observatory.preprocessing.cellwise import CellMaxRowsPreprocessor
from observatory.preprocessing.columnwise import ColumnwiseMaxRowsPreprocessor
from observatory.preprocessing.rowwise import RowwiseMaxRowsPreprocessor
from observatory.preprocessing.tablewise import TableMaxRowsPreprocessor


class TestRobertaEmbeddingInference(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "sample_data/wiki_tables",
        )
        model_name = "FacebookAI/roberta-base"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.wikitables_dataset = WikiTables(data_dir)
        self.model_wrapper = RobertaModelWrapper(model_name, device)

        # Prprocessor for inferring column embeddings
        self.columnwise_max_rows_preprocessor = ColumnwiseMaxRowsPreprocessor(
            tokenizer=self.model_wrapper.tokenizer,
            max_input_size=self.model_wrapper.max_input_size,
        )

        # Preprocessor for inferring row embeddings
        self.rowwise_max_rows_preprocessor = RowwiseMaxRowsPreprocessor(
            tokenizer=self.model_wrapper.tokenizer,
            max_input_size=self.model_wrapper.max_input_size,
        )

        # Preprocessor for inferring table embeddings
        self.table_max_rows_preprocessor = TableMaxRowsPreprocessor(
            tokenizer=self.model_wrapper.tokenizer,
            max_input_size=self.model_wrapper.max_input_size,
        )

        # Preprocessor for inferring cell embeddings
        self.cell_max_rows_preprocessor = CellMaxRowsPreprocessor(
            tokenizer=self.model_wrapper.tokenizer,
            max_input_size=self.model_wrapper.max_input_size,
        )

    def test_infer_column_embeddings_from_wikitables(self):
        truncated_tables = (
            self.columnwise_max_rows_preprocessor.truncate_columnwise(
                self.wikitables_dataset.all_tables
            )
        )
        column_embeddings = self.model_wrapper.infer_column_embeddings(
            truncated_tables, batch_size=32
        )
        assert len(column_embeddings) == len(self.wikitables_dataset.all_tables)

    def test_infer_row_embeddings_from_wikitables(self):
        truncated_tables = self.rowwise_max_rows_preprocessor.truncate_rowwise(
            self.wikitables_dataset.all_tables
        )
        row_embeddings = self.model_wrapper.infer_row_embeddings(
            truncated_tables, batch_size=32
        )
        assert len(row_embeddings) == len(self.wikitables_dataset.all_tables)

    def test_infer_table_embeddings_from_wikitables(self):
        # Serialize tables by column to infer table embeddings
        truncated_tables = self.table_max_rows_preprocessor.truncate_columnwise(
            self.wikitables_dataset.all_tables
        )
        table_embeddings = self.model_wrapper.infer_table_embeddings(
            truncated_tables, serialize_by_row=False, batch_size=32
        )
        assert len(table_embeddings) == len(self.wikitables_dataset.all_tables)

        # Serialize tables by row to infer table embeddings
        truncated_tables = self.table_max_rows_preprocessor.truncate_rowwise(
            self.wikitables_dataset.all_tables
        )
        table_embeddings = self.model_wrapper.infer_table_embeddings(
            truncated_tables, serialize_by_row=True, batch_size=32
        )
        assert len(table_embeddings) == len(self.wikitables_dataset.all_tables)

    def test_infer_cell_embeddings_from_wikitables(self):
        def assert_num_cells_matches_num_embeddings(
            tables: list[pd.DataFrame], cell_embeddings: list[torch.Tensor]
        ):
            """Assert the number of cells in each table matches the number of
            cell embeddings."""
            for i, tbl in enumerate(tables):
                num_cells = tbl.shape[0] * tbl.shape[1]
                assert len(cell_embeddings[i]) == num_cells

        # Include headers in serialized tables
        include_headers = True

        # Serialize tables by column to infer cell embeddings
        by_row = False
        truncated_tables = self.cell_max_rows_preprocessor.truncate(
            self.wikitables_dataset.all_tables, by_row, include_headers
        )
        cell_embeddings = self.model_wrapper.infer_cell_embeddings(
            truncated_tables,
            serialize_by_row=by_row,
            include_headers=include_headers,
            batch_size=32,
        )

        assert_num_cells_matches_num_embeddings(
            truncated_tables, cell_embeddings
        )

        # Serialize tables by row to infer cell embeddings
        by_row = True
        truncated_tables = self.cell_max_rows_preprocessor.truncate(
            self.wikitables_dataset.all_tables, by_row, include_headers
        )
        cell_embeddings = self.model_wrapper.infer_cell_embeddings(
            truncated_tables,
            serialize_by_row=by_row,
            include_headers=include_headers,
            batch_size=32,
        )

        assert_num_cells_matches_num_embeddings(
            truncated_tables, cell_embeddings
        )

        # Do not include headers in serialized tables
        include_headers = False

        # Serialize tables by column to infer cell embeddings
        by_row = False
        truncated_tables = self.cell_max_rows_preprocessor.truncate(
            self.wikitables_dataset.all_tables, by_row, include_headers
        )
        cell_embeddings = self.model_wrapper.infer_cell_embeddings(
            truncated_tables,
            serialize_by_row=by_row,
            include_headers=include_headers,
            batch_size=32,
        )

        assert_num_cells_matches_num_embeddings(
            truncated_tables, cell_embeddings
        )

        # Serialize tables by row to infer cell embeddings
        by_row = True
        truncated_tables = self.cell_max_rows_preprocessor.truncate(
            self.wikitables_dataset.all_tables, by_row, include_headers
        )
        cell_embeddings = self.model_wrapper.infer_cell_embeddings(
            truncated_tables,
            serialize_by_row=by_row,
            include_headers=include_headers,
            batch_size=32,
        )

        assert_num_cells_matches_num_embeddings(
            truncated_tables, cell_embeddings
        )
