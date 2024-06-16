import os
import unittest

import torch

from observatory.datasets.wikitables import WikiTables
from observatory.models.bert import BERTModelWrapper
from observatory.preprocessing.columnwise import ColumnwiseMaxRowsPreprocessor
from observatory.preprocessing.rowwise import RowwiseMaxRowsPreprocessor
from observatory.preprocessing.tablewise import TableMaxRowsPreprocessor


class TestBERTEmbeddingInference(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "sample_data/wiki_tables",
        )
        model_name = "bert-base-uncased"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.wikitables_dataset = WikiTables(data_dir)
        self.model_wrapper = BERTModelWrapper(model_name, device)

        self.columnwise_max_rows_preprocessor = ColumnwiseMaxRowsPreprocessor(
            tokenizer=self.model_wrapper.tokenizer,
            max_input_size=self.model_wrapper.max_input_size,
        )
        self.rowwise_max_rows_preprocessor = RowwiseMaxRowsPreprocessor(
            tokenizer=self.model_wrapper.tokenizer,
            max_input_size=self.model_wrapper.max_input_size,
        )
        self.table_max_rows_preprocessor = TableMaxRowsPreprocessor(
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

    def test_infer_column_embeddings_from_sotab(self):
        pass
