"""Deep RUL inference helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from aeroguard.deep.sequence_dataset import InferenceSequenceDataset, SequenceWindowDataset


@torch.no_grad()
def predict_batches(model: torch.nn.Module, dataset: SequenceWindowDataset | InferenceSequenceDataset, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    outputs = []
    for batch in loader:
        if len(batch) == 3:
            x, _, lengths = batch
        else:
            x, lengths = batch
        x = x.to(device)
        lengths = lengths.to(device)
        pred = model(x, lengths).detach().cpu().numpy().ravel()
        outputs.append(pred)
    if not outputs:
        return np.array([], dtype=float)
    return np.maximum(0.0, np.concatenate(outputs).astype(float))
