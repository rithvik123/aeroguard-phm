from __future__ import annotations

import torch
from torch import nn

from aeroguard.deep.models.common import PositiveHead, masked_mean, split_features_mask


class SequenceMLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.1, **_: object) -> None:
        super().__init__()
        feature_dim = input_dim - 1
        self.body = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = PositiveHead(hidden_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        features, mask = split_features_mask(x)
        pooled = masked_mean(features, mask)
        last = features[:, -1, :]
        return self.head(self.body(torch.cat([pooled, last], dim=1)))

