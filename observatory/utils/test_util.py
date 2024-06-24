import pandas as pd
import torch


def assert_num_embeddings_matches_number_columns(
    tables: list[pd.DataFrame],
    embeddings: torch.FloatTensor,
):
    """Assert the number of embeddings matches the number of columns in each
    table.

    Args:
        tables:
            A list of tables.
        embeddings:
            Column embeddings of shape (<num_embeddings>, <embedding size>).
    """

    num_columns = 0

    for tbl in tables:
        num_columns += tbl.shape[1]

    assert embeddings.shape[0] == num_columns
