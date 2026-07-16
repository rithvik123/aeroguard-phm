from __future__ import annotations

import torch
from torch import nn

from aeroguard.deep.models.common import CausalConv1d, PositiveHead, split_features_mask


class CNN1DRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1, kernel_size: int = 3, **_: object) -> None:
        super().__init__()
        channels = input_dim - 1
        self.receptive_field = 1 + 2 * (kernel_size - 1)
        self.net = nn.Sequential(
            CausalConv1d(channels, hidden_dim, kernel_size),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = PositiveHead(hidden_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        features, mask = split_features_mask(x)
        y = self.net(features.transpose(1, 2)).transpose(1, 2)
        pooled = (y * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)

