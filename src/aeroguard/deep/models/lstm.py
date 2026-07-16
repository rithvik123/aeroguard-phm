from __future__ import annotations

import torch
from torch import nn

from aeroguard.deep.models.common import PositiveHead, split_features_mask


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 80, layers: int = 1, dropout: float = 0.1, **_: object) -> None:
        super().__init__()
        recurrent_dropout = dropout if layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim - 1, hidden_dim, num_layers=layers, dropout=recurrent_dropout, batch_first=True, bidirectional=False)
        self.head = PositiveHead(hidden_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        features, _ = split_features_mask(x)
        output, _ = self.lstm(features)
        return self.head(output[:, -1, :])

