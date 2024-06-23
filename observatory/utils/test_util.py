from typing import Union

import pandas as pd
import torch


def assert_num_embeddings_matches_number_columns(
    tables: list[pd.DataFrame],
    embeddings: Union[torch.FloatTensor, list[torch.FloatTensor]],
):
    """Assert the number of embeddings matches the number of columns in each
    table.

    Args:
        tables:
            A list of tables.
        embeddings:
            Column embeddings per table or column embeddings in a single tensor.
    """

    if isinstance(embeddings, torch.FloatTensor):
        raise NotImplementedError
    else:
        for i, tbl in enumerate(tables):
            assert embeddings[i].shape[0] == tbl.shape[1]
