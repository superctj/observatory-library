# Observatory
A Python library for embedding inference of relational tabular data. This repository evolves from the [codebase](https://github.com/superctj/observatory/tree/main) of our VLDB 2024 paper [Observatory: Characterizing Embeddings of Relational Tables](https://www.vldb.org/pvldb/vol17/p849-cong.pdf).

We are open-sourcing Observatory library for beta-test. The library is under active development and we welcome feedback and contributions. Please feel free to open an issue or submit a pull request.

## Installation

### Install from source
Assume using [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) for Python package management on Linux machines.

1. Clone the repository and go to the project directory:
    ```
    git clone <repo url>
    cd observatory-library
    ```

2. Create and activate the environment:
    ```
    conda env create -f cpu_environment.yml
    conda activate observatory
    ```

    If you have access to GPUs, install the GPU environment:
    ```
    conda env create -f cuda_11.8_environment.yml
    conda activate observatory
    ```

    We will provide the environment file for cuda 12.x soon.


### Install from Conda
Coming soon after beta-test.


## Quick Start
```python
import os

import torch

from observatory.datasets.sotab import SotabDataset, collate_fn
from observatory.models.bert_family import BertModelWrapper
from observatory.preprocessing.columnwise import (
    ColumnwiseCellDocumentFrequencyBasedPreprocessor,
)
from torch.utils.data import DataLoader

# Initialize data (the metadata file simply lists all the table file names)
data_dir = "./tests/sample_data/wiki_tables"
metadata_filepath = os.path.join(data_dir, "table_inventory.csv")
sotab_dataset = SotabDataset(data_dir, metadata_filepath)

sotab_dataloader = DataLoader(
    sotab_dataset,
    batch_size=4,  # batch size for loading tables
    shuffle=False,
    collate_fn=collate_fn,
)

# Initialize model
model_name = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_wrapper = BertModelWrapper(model_name, device)

# Create cell document frequency-based preprocessor for inferring column embeddings
cell_frequencies = sotab_dataset.compute_cell_document_frequencies()
columnwise_preprocessor = ColumnwiseCellDocumentFrequencyBasedPreprocessor(
    tokenizer=model_wrapper.tokenizer,
    max_input_size=model_wrapper.max_input_size,
    cell_frequencies=cell_frequencies,
    include_table_name=True,
    include_column_names=True,
    include_column_stats=True,
)

# Infer column embeddings
for batch_tables in sotab_dataloader:
    encoded_inputs, _ = columnwise_preprocessor.serialize(batch_tables)

    column_embeddings = model_wrapper.batch_infer_embeddings(
        encoded_inputs, batch_size=16  # batch size for embedding inference
    )
```

You can find more examples of embedding inference in the `tests` directory.

## Features

### Leave serialization, encoding, and (batch) inference to Observatory

We currently support the following preprocessors:

| Preprocessor                           | Embedding Inference      | Source       |
| :------------------------------------- | :----------------------: | :----------- |
| CellDocumentFrequencyBasedPreprocessor | column                   | [DeepJoin](https://www.vldb.org/pvldb/vol16/p2458-dong.pdf) |
| MaxRowsPreprocessor                    | column, row, table, cell | [Observatory](https://www.vldb.org/pvldb/vol17/p849-cong.pdf) |


### Easy integration with Hugging Face models
We currently support any BERT-like encoder model including BERT, RoBERTa and ALBERT. To extend the support to other models, simply implement a wrapper class that inherits from `BERTFamilyModelWrapper` and implements the `get_model` method. For example,

```
from transformers import AlbertModel

class AlbertModelWrapper(BERTFamilyModelWrapper):
    def get_model(self) -> AlbertModel:
        try:
            model = AlbertModel.from_pretrained(
                self.model_name, local_files_only=True
            )
        except OSError:
            model = AlbertModel.from_pretrained(self.model_name)

        model = model.to(self.device)
        model.eval()

        return model
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
