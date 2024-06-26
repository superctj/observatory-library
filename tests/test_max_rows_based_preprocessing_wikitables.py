import os
import unittest

import torch

from torch.utils.data import DataLoader

from observatory.datasets.wikitables import WikiTablesDataset, collate_fn
from observatory.models.bert_family import BertModelWrapper
from observatory.preprocessing.cellwise import CellwiseMaxRowsPreprocessor
from observatory.preprocessing.columnwise import ColumnwiseMaxRowsPreprocessor
from observatory.preprocessing.rowwise import RowwiseMaxRowsPreprocessor
from observatory.preprocessing.tablewise import TablewiseMaxRowsPreprocessor
from observatory.utils.test_util import (
    assert_num_embeddings_matches_num_cells,
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
        for by_row in [True, False]:
            tablewise_max_rows_preprocessor = TablewiseMaxRowsPreprocessor(
                tokenizer=self.model_wrapper.tokenizer,
                max_input_size=self.model_wrapper.max_input_size,
                by_row=by_row,
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

                # Passing in positions of [CLS] tokens should yield the same
                # results
                same_table_embeddings = self.model_wrapper.infer_embeddings(
                    encoded_inputs, cls_positions
                )

                assert torch.equal(table_embeddings, same_table_embeddings)

    def test_infer_cell_embeddings(self):
        # Max rows-based preprocessor for inferring cell embeddings
        for by_row in [True, False]:
            # Setting `include_table_name=True` will yield a table-too-wide
            # error. This is expected as max rows-based preprocessor expects at
            # least one row of a table fits within the maximum model input size
            cellwise_max_rows_preprocessor = CellwiseMaxRowsPreprocessor(
                tokenizer=self.model_wrapper.tokenizer,
                max_input_size=self.model_wrapper.max_input_size,
                by_row=by_row,
                include_table_name=False,
                include_column_names=True,
            )

            for batch_tables in self.wikitables_dataloader:
                encoded_inputs, cell_positions = (
                    cellwise_max_rows_preprocessor.serialize(batch_tables)
                )

                cell_embeddings = (
                    self.model_wrapper.infer_embeddings_by_averaging_tokens(
                        encoded_inputs, cell_positions
                    )
                )

                assert_num_embeddings_matches_num_cells(
                    cell_embeddings, cell_token_spans=cell_positions
                )


class TestRobertaEmbeddingInference(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "sample_data/wiki_tables",
        )
        metadata_filepath = os.path.join(data_dir, "table_inventory.csv")
        wikitables_dataset = WikiTablesDataset(data_dir, metadata_filepath)

        model_name = "FacebookAI/roberta-base"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_wrapper = BertModelWrapper(model_name, device)
        self.wikitables_dataloader = DataLoader(
            wikitables_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn,
        )

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
        for by_row in [True, False]:
            tablewise_max_rows_preprocessor = TablewiseMaxRowsPreprocessor(
                tokenizer=self.model_wrapper.tokenizer,
                max_input_size=self.model_wrapper.max_input_size,
                by_row=by_row,
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

                # Passing in positions of [CLS] tokens should yield the same
                # results
                same_table_embeddings = self.model_wrapper.infer_embeddings(
                    encoded_inputs, cls_positions
                )

                assert torch.equal(table_embeddings, same_table_embeddings)

    def test_infer_cell_embeddings(self):
        # Max rows-based preprocessor for inferring cell embeddings
        for by_row in [True, False]:
            # Setting `include_table_name=True` will yield a table-too-wide
            # error. This is expected as max rows-based preprocessor expects at
            # least one row of a table fits within the maximum model input size
            cellwise_max_rows_preprocessor = CellwiseMaxRowsPreprocessor(
                tokenizer=self.model_wrapper.tokenizer,
                max_input_size=self.model_wrapper.max_input_size,
                by_row=by_row,
                include_table_name=False,
                include_column_names=True,
            )

            for batch_tables in self.wikitables_dataloader:
                encoded_inputs, cell_positions = (
                    cellwise_max_rows_preprocessor.serialize(batch_tables)
                )

                cell_embeddings = (
                    self.model_wrapper.infer_embeddings_by_averaging_tokens(
                        encoded_inputs, cell_positions
                    )
                )

                assert_num_embeddings_matches_num_cells(
                    cell_embeddings, cell_token_spans=cell_positions
                )


class TestAlbertEmbeddingInference(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "sample_data/wiki_tables",
        )
        metadata_filepath = os.path.join(data_dir, "table_inventory.csv")
        wikitables_dataset = WikiTablesDataset(data_dir, metadata_filepath)

        model_name = "albert/albert-base-v2"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_wrapper = BertModelWrapper(model_name, device)
        self.wikitables_dataloader = DataLoader(
            wikitables_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn,
        )

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
        for by_row in [True, False]:
            tablewise_max_rows_preprocessor = TablewiseMaxRowsPreprocessor(
                tokenizer=self.model_wrapper.tokenizer,
                max_input_size=self.model_wrapper.max_input_size,
                by_row=by_row,
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

                # Passing in positions of [CLS] tokens should yield the same
                # results
                same_table_embeddings = self.model_wrapper.infer_embeddings(
                    encoded_inputs, cls_positions
                )

                assert torch.equal(table_embeddings, same_table_embeddings)

    def test_infer_cell_embeddings(self):
        # Max rows-based preprocessor for inferring cell embeddings
        for by_row in [True, False]:
            # Setting `include_table_name=True` will yield a table-too-wide
            # error. This is expected as max rows-based preprocessor expects at
            # least one row of a table fits within the maximum model input size
            cellwise_max_rows_preprocessor = CellwiseMaxRowsPreprocessor(
                tokenizer=self.model_wrapper.tokenizer,
                max_input_size=self.model_wrapper.max_input_size,
                by_row=by_row,
                include_table_name=False,
                include_column_names=True,
            )

            for batch_tables in self.wikitables_dataloader:
                encoded_inputs, cell_positions = (
                    cellwise_max_rows_preprocessor.serialize(batch_tables)
                )

                cell_embeddings = (
                    self.model_wrapper.infer_embeddings_by_averaging_tokens(
                        encoded_inputs, cell_positions
                    )
                )

                assert_num_embeddings_matches_num_cells(
                    cell_embeddings, cell_token_spans=cell_positions
                )
