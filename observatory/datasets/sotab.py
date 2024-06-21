import os

import pandas as pd


class SotabDataset:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # self.all_tables = self._get_tables()

    # def _get_tables(self):
    #     all_tables = []

    #     for f in os.listdir(self.data_dir):
    #         if f.endswith(".json.gz"):
    #             filepath = os.path.join(self.data_dir, f)
    #             table = pd.read_json(filepath, compression="gzip", lines=True)

    #             table.columns = table.columns.astype(str)
    #             table = table.reset_index(drop=True)
    #             table = table.astype(str)

    #             all_tables.append(table)

    #     return all_tables

    def compute_cell_document_frequencies(self):
        """Compute the document frequency of each cell value, which is defined
        as the number of columns in the table corpus that have the cell value.

        Returns:
            A dictionary mapping cell values to their document frequencies.
        """

        cell_document_frequencies = {}

        for f in os.listdir(self.data_dir):
            if f.endswith(".json.gz"):
                filepath = os.path.join(self.data_dir, f)
                table = pd.read_json(filepath, compression="gzip", lines=True)

                # Deop NaN columns and rows
                table.dropna(axis=1, how="all", inplace=True)
                table.dropna(axis=0, how="all", inplace=True)

                # Convert all columns to string
                table.columns = table.columns.astype(str)
                table = table.astype(str)

                for col in table.columns:
                    for cell in table[col]:
                        if cell not in cell_document_frequencies:
                            cell_document_frequencies[cell] = 1
                        else:
                            cell_document_frequencies[cell] += 1

        return cell_document_frequencies
