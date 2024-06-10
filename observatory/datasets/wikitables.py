import os

import pandas as pd


class WikiTables:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.all_tables = self._get_tables()

    def _get_tables(self):
        all_tables = []

        for f in os.listdir(self.data_dir):
            if f.endswith(".csv"):
                table = pd.read_csv(os.path.join(self.data_dir, f))

                table.columns = table.columns.astype(str)
                table = table.reset_index(drop=True)
                table = table.astype(str)

                all_tables.append(table)

        return all_tables
