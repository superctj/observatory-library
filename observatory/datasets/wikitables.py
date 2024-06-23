import os

import pandas as pd

from torch.utils.data import Dataset


class WikiTablesDataset(Dataset):
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
        self.transform = transform

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, idx) -> pd.DataFrame:
        # Assume the first column of the metadata file contains table file names
        table_filepath = os.path.join(self.data_dir, self.metadata.iloc[idx, 0])
        table = pd.read_csv(table_filepath)

        # Add table name
        table.attrs["name"] = self.metadata.iloc[idx, 0]

        # Drop NaN columns and rows
        table.dropna(axis=1, how="all", inplace=True)
        table.dropna(axis=0, how="all", inplace=True)

        # Convert all columns to string
        table = table.astype(str)

        if self.transform:
            table = self.transform(table)

        return table


def collate_fn(batch: list[pd.DataFrame]):
    return batch
