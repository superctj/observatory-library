from abc import ABC, abstractmethod

from transformers import PreTrainedTokenizer


class PreprocessingWrapper(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_input_size: int):
        self.tokenizer = tokenizer
        self.max_input_size = max_input_size

    @abstractmethod
    def is_fit(self):
        pass
