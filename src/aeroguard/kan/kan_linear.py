"""Edge-function KAN linear layer."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from aeroguard.kan.spline_basis import bspline_basis


class KANLinear(nn.Module):
    """A compact edge-function KAN layer.

    Each input-output edge learns ``w_base * SiLU(x_i) + sum_k c_k B_k(x_i)``.
    The layer sums edge contributions across inputs for each output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_size: int = 5,
        spline_degree: int = 3,
        input_clamp: float = 5.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(int(seed))
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_degree = int(spline_degree)
        self.input_clamp = float(input_clamp)
        self.n_basis = self.grid_size + self.spline_degree
        self.base_weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.spline_coeff = nn.Parameter(torch.empty(self.out_features, self.in_features, self.n_basis))
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        self.register_buffer("edge_mask", torch.ones(self.out_features, self.in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(max(1, self.in_features))
        nn.init.uniform_(self.base_weight, -bound, bound)
        nn.init.normal_(self.spline_coeff, mean=0.0, std=0.01)
        nn.init.zeros_(self.bias)

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        return bspline_basis(x, grid_size=self.grid_size, degree=self.spline_degree, value_min=-self.input_clamp, value_max=self.input_clamp)

    def edge_contributions(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = torch.clamp(x, -self.input_clamp, self.input_clamp)
        base = F.silu(x_clamped).unsqueeze(1) * self.base_weight.unsqueeze(0)
        basis = self.basis(x_clamped)
        spline = torch.einsum("bik,oik->boi", basis, self.spline_coeff)
        return (base + spline) * self.edge_mask.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.edge_contributions(x).sum(dim=2) + self.bias

    def edge_importance(self) -> torch.Tensor:
        spline_magnitude = self.spline_coeff.detach().abs().mean(dim=2)
        return (self.base_weight.detach().abs() + spline_magnitude) * self.edge_mask.detach()

    def parameter_count(self, *, active_only: bool = False) -> int:
        if not active_only:
            return sum(parameter.numel() for parameter in self.parameters())
        active_edges = int(self.edge_mask.detach().sum().item())
        return active_edges * (1 + self.n_basis) + self.out_features

    def apply_pruning_mask(self, mask: torch.Tensor) -> None:
        if mask.shape != self.edge_mask.shape:
            raise ValueError(f"Mask shape {tuple(mask.shape)} does not match edge mask {tuple(self.edge_mask.shape)}")
        self.edge_mask.copy_(mask.to(device=self.edge_mask.device, dtype=self.edge_mask.dtype))
