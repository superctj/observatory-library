import os
import unittest

import torch

from torch.utils.data import DataLoader

from observatory.datasets.sotab import SotabDataset
from observatory.models.bert_family import BertModelWrapper
from observatory.preprocessing.columnwise import (
    ColumnwiseDocumentFrequencyBasedPreprocessor,
)


class TestBertEmbeddingInference(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "sample_data/sotab",
        )
        metadata_filepath = os.path.join(data_dir, "table_inventory.csv")

        model_name = "google-bert/bert-base-uncased"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sotab_dataset = SotabDataset(data_dir, metadata_filepath)
        cell_frequencies = (
            self.sotab_dataset.compute_cell_document_frequencies()
        )
        # self.sotab_dataloader = DataLoader(
        #     sotab_dataset, batch_size=1, shuffle=False
        # )
        self.model_wrapper = BertModelWrapper(model_name, device)

        # Preprocessor for inferring column embeddings
        self.columnwise_preprocessor = (
            ColumnwiseDocumentFrequencyBasedPreprocessor(
                tokenizer=self.model_wrapper.tokenizer,
                max_input_size=self.model_wrapper.max_input_size,
                cell_frequencies=cell_frequencies,
                include_table_name=True,
                include_column_name=True,
                include_column_stats=True,
            )
        )

    def test_infer_column_embeddings(self):
        for table in self.sotab_dataset:
            encoded_inputs = self.columnwise_preprocessor.serialize_columnwise(
                table
            )

            column_embeddings = self.model_wrapper.infer_embeddings(
                encoded_inputs
            )

            assert column_embeddings.shape[0] == len(table.columns)

    def test_infer_row_embeddings(self):
        pass

    def test_infer_table_embeddings(self):
        pass

    def test_infer_cell_embeddings(self):
        pass
