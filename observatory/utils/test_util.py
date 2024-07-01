from typing import Union

import pandas as pd
import torch


def assert_num_embeddings_matches_num_columns(
    embeddings: torch.FloatTensor,
    num_columns: int = None,
    tables: list[pd.DataFrame] = None,
):
    """Assert the number of embeddings matches the total number of columns.

    Args:
        embeddings:
            Column embeddings of shape (<num_embeddings>, <embedding size>).
        num_columns (optional):
            Number of columns.
        tables (optional):
            A list of tables.
    """

    if num_columns:
        assert embeddings.shape[0] == num_columns
    else:
        assert tables is not None

        num_columns = 0
        for tbl in tables:
            num_columns += tbl.shape[1]

        assert embeddings.shape[0] == num_columns


def assert_num_embeddings_matches_num_rows(
    cls_positions: list[list[int]],
    embeddings: torch.FloatTensor,
):
    """Assert the number of embeddings matches the total number of rows.

    Args:
        cls_positions:
            Lists of positions of [CLS] tokens.
        embeddings:
            Row embeddings of shape (<num_embeddings>, <embedding size>).
    """

    num_rows = 0

    for cls_pos in cls_positions:
        num_rows += len(cls_pos)

    assert embeddings.shape[0] == num_rows


def assert_num_embeddings_matches_num_tables(
    tables: Union[list[pd.DataFrame], list[list[int]]],
    embeddings: torch.FloatTensor,
):
    """Assert the number of embeddings matches the total number of tables.

    Args:
        tables:
            A list of tables or lists of positions of [CLS] tokens.
        embeddings:
            Table embeddings of shape (<num_embeddings>, <embedding size>).
    """

    assert embeddings.shape[0] == len(tables)


def assert_num_embeddings_matches_num_cells(
    cell_embeddings: list[torch.FloatTensor],
    tables: list[pd.DataFrame] = None,
    cell_token_spans: list[list[tuple[int]]] = None,
):
    """Assert the number of cell embeddings matches the number of cells in each
    table.

    Args:
        cell_embeddings:
            Cell embeddings of shape (<num_embeddings>, <embedding size>).
        tables:
            A list of tables.
        cell_token_spans:
            Lists of cell token spans per table.
    """

    if tables:
        for i, tbl in enumerate(tables):
            assert cell_embeddings[i].shape[0] == tbl.size
    else:
        assert cell_token_spans is not None

        for i, spans in enumerate(cell_token_spans):
            assert cell_embeddings[i].shape[0] == len(spans)
