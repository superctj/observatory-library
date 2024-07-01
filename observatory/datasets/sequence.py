import torch

from torch.utils.data import Dataset


class EncodedInputsDataset(Dataset):
    def __init__(self, encoded_inputs: dict, cls_positions: list[list[int]]):
        self.encoded_inputs = encoded_inputs
        self.cls_positions = cls_positions

    def __len__(self):
        return self.encoded_inputs["input_ids"].shape[0]

    def __getitem__(self, index) -> tuple[dict, list[int]]:
        return (
            {key: value[index] for key, value in self.encoded_inputs.items()},
            self.cls_positions[index],
        )


def encoded_inputs_collate_fn(batch: tuple[dict, list[int]]):
    encoded_inputs, cls_positions = zip(*batch)

    return {
        key: torch.stack([inputs[key] for inputs in encoded_inputs])
        for key in encoded_inputs[0].keys()
    }, cls_positions
