from __future__ import annotations

import torch
from torch import nn

from aeroguard.deep.models.common import CausalConv1d, PositiveHead, split_features_mask


class CNNLSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 72, dropout: float = 0.1, kernel_size: int = 3, layers: int = 1, **_: object) -> None:
        super().__init__()
        self.conv = nn.Sequential(CausalConv1d(input_dim - 1, hidden_dim, kernel_size), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=layers, batch_first=True, bidirectional=False)
        self.head = PositiveHead(hidden_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        features, _ = split_features_mask(x)
        y = self.conv(features.transpose(1, 2)).transpose(1, 2)
        output, _ = self.lstm(y)
        return self.head(output[:, -1, :])

