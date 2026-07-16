from __future__ import annotations

import torch
from torch import nn

from aeroguard.deep.models.common import CausalConv1d, PositiveHead, split_features_mask


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.net = nn.Sequential(self.conv1, nn.GELU(), nn.Dropout(dropout), self.conv2, nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1, kernel_size: int = 3, dilations: list[int] | None = None, **_: object) -> None:
        super().__init__()
        self.dilations = dilations or [1, 2, 4]
        self.receptive_field = 1 + 2 * sum((kernel_size - 1) * dilation for dilation in self.dilations)
        self.input = nn.Conv1d(input_dim - 1, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(*[TCNBlock(hidden_dim, kernel_size, dilation, dropout) for dilation in self.dilations])
        self.head = PositiveHead(hidden_dim, hidden_dim, dropout)

    def sequence_features(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = split_features_mask(x)
        y = self.input(features.transpose(1, 2))
        return self.blocks(y).transpose(1, 2)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        y = self.sequence_features(x)
        return self.head(y[:, -1, :])

