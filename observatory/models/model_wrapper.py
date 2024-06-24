import logging

from abc import ABC, abstractmethod

import torch


class ModelWrapper(ABC):
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.model = self.get_model()
        self.tokenizer = self.get_tokenizer()
        self.max_input_size = self.get_max_input_size()

        assert self.tokenizer.cls_token is not None
        assert self.tokenizer.sep_token is not None
        assert self.tokenizer.pad_token is not None
        assert self.max_input_size is not None

    @abstractmethod
    def get_model(self):
        """Get a pretrained model."""
        pass

    @abstractmethod
    def get_tokenizer(self):
        """Get the corresponding tokenizer."""
        pass

    @abstractmethod
    def get_max_input_size(self):
        """Get the maximum input size for the model.

        As there is no reliable way to get this information from Hugging Face
        models, we currently hard-code this value for each model.
        """
        pass

    @abstractmethod
    def infer_embeddings(self):
        pass

    @abstractmethod
    def batch_infer_embeddings(self):
        pass
