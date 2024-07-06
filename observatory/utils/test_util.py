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
    embeddings: torch.FloatTensor,
    num_rows: int = None,
    cls_positions: list[list[int]] = None,
):
    """Assert the number of embeddings matches the total number of rows.

    Args:
        embeddings:
            Row embeddings of shape (<num_embeddings>, <embedding size>).
        cls_positions:
            Lists of positions of [CLS] tokens, each of which represents a row.
    """

    if num_rows:
        assert embeddings.shape[0] == num_rows
    else:
        assert cls_positions is not None

        num_rows = 0
        for cls_pos in cls_positions:
            num_rows += len(cls_pos)

        assert embeddings.shape[0] == num_rows


def assert_num_embeddings_matches_num_tables(
    embeddings: torch.FloatTensor, num_tables: int
):
    """Assert the number of embeddings matches the total number of tables.

    Args:
        embeddings:
            Table embeddings of shape (<num_embeddings>, <embedding size>).
        num_tables:
            Number of tables.
    """

    assert embeddings.shape[0] == num_tables


def assert_num_embeddings_matches_num_cells(
    cell_embeddings: list[torch.FloatTensor],
    cell_token_spans: list[list[tuple[int]]],
):
    """Assert the number of cell embeddings matches the number of cells in each
    table.

    Args:
        cell_embeddings:
            Cell embeddings of shape (<num_embeddings>, <embedding size>).
        cell_token_spans:
            Lists of cell token spans per table.
    """

    for i, spans in enumerate(cell_token_spans):
        assert cell_embeddings[i].shape[0] == len(spans)
