import pandas as pd
import torch


def assert_num_embeddings_matches_number_columns(
    tables: list[pd.DataFrame],
    embeddings: torch.FloatTensor,
):
    """Assert the number of embeddings matches the total number of columns.

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


def assert_num_embeddings_matches_number_rows(
    cls_positions: list[list[int]],
    embeddings: torch.FloatTensor,
):
    """Assert the number of embeddings matches the total number of rows.

    Args:
        cls_positions:
            Lists of positions of [CLS] tokens.
        embeddings:
            Column embeddings of shape (<num_embeddings>, <embedding size>).
    """

    num_rows = 0

    for cls_pos in cls_positions:
        num_rows += len(cls_pos)

    assert embeddings.shape[0] == num_rows
