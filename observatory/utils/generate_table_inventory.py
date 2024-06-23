import argparse
import os

import pandas as pd


def generate_table_inventory(
    data_dir: str, file_extension: str
) -> pd.DataFrame:
    """Generate an inventory of tables in a directory.

    Args:
        data_dir: The directory containing tables.
        file_extension: The file extension of the tables.

    Returns:
        A DataFrame containing the inventory of tables.
    """

    table_inventory = {"table_name": []}

    for f in os.listdir(data_dir):
        if f.endswith(file_extension):
            table_inventory["table_name"].append(f)

    return pd.DataFrame(table_inventory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        help="The directory containing tables.",
    )

    parser.add_argument(
        "--file_extension",
        type=str,
        default=".csv",
        help="The table file extension.",
    )

    args = parser.parse_args()

    table_inventory = generate_table_inventory(
        args.data_dir,
        args.file_extension,
    )

    table_inventory.to_csv(
        os.path.join(args.data_dir, "table_inventory.csv"), index=False
    )
