"""Regularization helpers for local KAN layers."""

from __future__ import annotations

import torch
from torch import nn

from aeroguard.kan.kan_linear import KANLinear


def edge_sparsity_penalty(module: nn.Module) -> torch.Tensor:
    penalties = []
    for layer in module.modules():
        if isinstance(layer, KANLinear):
            penalties.append(layer.base_weight.abs().mean() + layer.spline_coeff.abs().mean())
    if not penalties:
        return torch.tensor(0.0)
    return torch.stack(penalties).sum()


def spline_smoothness_penalty(module: nn.Module) -> torch.Tensor:
    penalties = []
    for layer in module.modules():
        if isinstance(layer, KANLinear) and layer.spline_coeff.shape[-1] > 2:
            second_diff = layer.spline_coeff[..., 2:] - 2.0 * layer.spline_coeff[..., 1:-1] + layer.spline_coeff[..., :-2]
            penalties.append(second_diff.pow(2).mean())
    if not penalties:
        return torch.tensor(0.0)
    return torch.stack(penalties).sum()
