"""Common neural-network blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def split_features_mask(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = x[..., :-1]
    mask = x[..., -1:].clamp(0.0, 1.0)
    return features, mask


def masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (features * mask).sum(dim=1) / denom


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


class PositiveHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        if kernel_size <= 0 or dilation <= 0:
            raise ValueError("kernel_size and dilation must be positive.")
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


def validate_parameter_budget(model: nn.Module, budget: int) -> None:
    count = trainable_parameter_count(model)
    if count > int(budget):
        raise ValueError(f"Model has {count} trainable parameters, exceeding budget {budget}.")

