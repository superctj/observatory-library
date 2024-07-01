import os
import unittest

import torch

from torch.utils.data import DataLoader

from observatory.datasets.sequence import (
    EncodedInputsDataset,
    encoded_inputs_collate_fn,
)
from observatory.datasets.sotab import SotabDataset, collate_fn
from observatory.models.bert_family import BertModelWrapper
from observatory.preprocessing.columnwise import (
    ColumnwiseCellDocumentFrequencyBasedPreprocessor,
)
from observatory.utils.test_util import (
    assert_num_embeddings_matches_num_columns,
)


class TestBertEmbeddingInference(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "sample_data/sotab",
        )
        metadata_filepath = os.path.join(data_dir, "table_inventory.csv")
        sotab_dataset = SotabDataset(data_dir, metadata_filepath)

        # Compute cell document frequencies
        self.cell_frequencies = (
            sotab_dataset.compute_cell_document_frequencies()
        )

        # The preprocessor below serializes each column to a sequence as input
        # to the model, so `inference_batch_size` can be different from
        # `table_batch_size` used in the table data loader.
        table_batch_size = 4
        self.inference_batch_size = 16

        self.sotab_dataloader = DataLoader(
            sotab_dataset,
            batch_size=table_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model_name = "google-bert/bert-base-uncased"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_wrapper = BertModelWrapper(model_name, device)

    def test_infer_column_embeddings(self):
        # Create cell document frequency-based preprocessor for inferring
        # column embeddings
        columnwise_preprocessor = (
            ColumnwiseCellDocumentFrequencyBasedPreprocessor(
                tokenizer=self.model_wrapper.tokenizer,
                max_input_size=self.model_wrapper.max_input_size,
                cell_frequencies=self.cell_frequencies,
                include_table_name=True,
                include_column_names=True,
                include_column_stats=True,
            )
        )

        for batch_tables in self.sotab_dataloader:
            encoded_inputs, cls_positions = columnwise_preprocessor.serialize(
                batch_tables
            )

            encoded_inputs_dataset = EncodedInputsDataset(
                encoded_inputs, cls_positions
            )
            encoded_inputs_dataloader = DataLoader(
                encoded_inputs_dataset,
                batch_size=self.inference_batch_size,
                collate_fn=encoded_inputs_collate_fn,
            )

            for (
                batch_encoded_inputs,
                batch_cls_positions,
            ) in encoded_inputs_dataloader:
                column_embeddings = self.model_wrapper.infer_embeddings(
                    batch_encoded_inputs, batch_cls_positions
                )

                assert_num_embeddings_matches_num_columns(
                    column_embeddings, num_columns=len(batch_cls_positions)
                )

                assert_num_embeddings_matches_num_columns(
                    column_embeddings,
                    num_columns=batch_encoded_inputs["input_ids"].shape[0],
                )

    def test_infer_row_embeddings(self):
        pass

    def test_infer_table_embeddings(self):
        pass

    def test_infer_cell_embeddings(self):
        pass
