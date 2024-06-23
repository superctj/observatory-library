import os

import pandas as pd

from torch.utils.data import Dataset


class SotabDataset(Dataset):
    def __init__(
        self, data_dir: str, metadata_filepath: str, transform: callable = None
    ):
        """
        Args:
            data_dir:
                The directory containing tables.
            metadata_filepath:
                The path to the metadata file, which contains table file names
                and potentially other information of tables.
        """

        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_filepath)

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, idx) -> pd.DataFrame:
        # Assume the first column of the metadata file contains table file names
        table_filepath = os.path.join(self.data_dir, self.metadata.iloc[idx, 0])
        table = pd.read_json(table_filepath, compression="gzip", lines=True)

        # Add table name
        table.attrs["name"] = self.metadata.iloc[idx, 0]

        # Drop NaN columns and rows
        table.dropna(axis=1, how="all", inplace=True)
        table.dropna(axis=0, how="all", inplace=True)

        # Convert all columns to string
        table = table.astype(str)

        return table

    def compute_cell_document_frequencies(self):
        """Compute the document frequency of each cell value, which is defined
        as the number of columns in the table corpus that have the cell value.

        Returns:
            A dictionary mapping cell values to their document frequencies.
        """

        cell_document_frequencies = {}

        for row in self.metadata.itertuples():
            table = pd.read_json(
                os.path.join(self.data_dir, row.table_name),
                compression="gzip",
                lines=True,
            )

            # Drop NaN columns and rows
            table.dropna(axis=1, how="all", inplace=True)
            table.dropna(axis=0, how="all", inplace=True)

            # Convert all columns to string
            table = table.astype(str)

            for col in table.columns:
                col_uniq_values = table[col].unique()

                for cell in col_uniq_values:
                    if cell not in cell_document_frequencies:
                        cell_document_frequencies[cell] = 1
                    else:
                        cell_document_frequencies[cell] += 1

        return cell_document_frequencies
