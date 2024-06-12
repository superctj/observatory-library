# Observatory
A Python library for embedding inference of relational tabular data. This repository evolves from the [codebase](https://github.com/superctj/observatory/tree/main) of our VLDB 2024 paper.

## Installation


## Features


## Quick Start
```python
import torch

from observatory.datasets.wikitables import WikiTables
from observatory.models.bert import BERTModelWrapper
from observatory.preprocessing.columnwise import MaxRowsPreprocessor

# Initialize data
data_dir = ...
wikitables_dataset = WikiTables(data_dir)

# Initialize model
model_name = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_wrapper = BERTModelWrapper(model_name, device)

# Preprocess data
max_rows_preprocessor = MaxRowsPreprocessor(
    tokenizer=model_wrapper.tokenizer,
    max_input_size=model_wrapper.max_input_size,
)

truncated_tables = max_rows_preprocessor.columnwise_truncation(
    wikitables_dataset.all_tables
)

# Infer column embeddings
column_embeddings = model_wrapper.infer_column_embeddings(
    truncated_tables, batch_size=32
)
```


## Citing Observatory
If you find Observatory useful for your work, please cite the following BibTeX:

```bibtex
@article{cong2023observatory,
  author  = {Tianji Cong and
             Madelon Hulsebos and
             Zhenjie Sun and
             Paul Groth and
             H. V. Jagadish},
  title   = {Observatory: Characterizing Embeddings of Relational Tables},
  journal = {Proc. {VLDB} Endow.},
  volume  = {17},
  number  = {4},
  pages   = {849--862},
  year    = {2023},
}
```

```bibtex
@inproceedings{cong2023observatorylibrary,
  author    = {Cong, Tianji and
               Sun, Zhenjie and
               Groth, Paul and
               Jagadish, H. V. and
               Hulsebos, Madelon},
  title     = {Introducing the Observatory Library for End-to-End Table Embedding Inference},
  booktitle = {The 2nd Table Representation Learning Workshop at NeurIPS 2023},
  publisher = {https://table-representation-learning.github.io},
  year      = {2023}
}
```
