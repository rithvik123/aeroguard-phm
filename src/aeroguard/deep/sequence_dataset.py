"""Torch dataset wrappers for sequence windows."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceWindowDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, lengths: np.ndarray | None = None) -> None:
        if len(sequences) != len(targets):
            raise ValueError("sequences and targets must be aligned.")
        if len(sequences) == 0:
            raise ValueError("SequenceWindowDataset must not be empty.")
        if not np.isfinite(sequences).all() or not np.isfinite(targets).all():
            raise ValueError("SequenceWindowDataset inputs must be finite.")
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32).view(-1, 1)
        if lengths is None:
            lengths = sequences[:, :, -1].sum(axis=1)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)

    def __len__(self) -> int:
        return int(len(self.targets))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.targets[index], self.lengths[index]


class InferenceSequenceDataset(Dataset):
    """Label-free sequence windows for benchmark or production inference."""

    def __init__(self, sequences: np.ndarray, lengths: np.ndarray | None = None) -> None:
        if len(sequences) == 0:
            raise ValueError("InferenceSequenceDataset must not be empty.")
        if not np.isfinite(sequences).all():
            raise ValueError("InferenceSequenceDataset inputs must be finite.")
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        if lengths is None:
            lengths = sequences[:, :, -1].sum(axis=1)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)

    def __len__(self) -> int:
        return int(len(self.sequences))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.lengths[index]
