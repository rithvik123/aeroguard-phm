"""Sparse KAN regressors used by the Phase 5D residual corrector."""

from __future__ import annotations

import torch
from torch import nn

from aeroguard.kan.kan_linear import KANLinear


class SparseKANRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        grid_size: int = 5,
        spline_degree: int = 3,
        input_clamp: float = 5.0,
        hidden_nodes: int = 0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_nodes = int(hidden_nodes)
        if hidden_nodes > 0:
            self.first = KANLinear(input_dim, hidden_nodes, grid_size=grid_size, spline_degree=spline_degree, input_clamp=input_clamp, seed=seed)
            self.second = KANLinear(hidden_nodes, 1, grid_size=grid_size, spline_degree=spline_degree, input_clamp=input_clamp, seed=None if seed is None else seed + 1)
        else:
            self.first = KANLinear(input_dim, 1, grid_size=grid_size, spline_degree=spline_degree, input_clamp=input_clamp, seed=seed)
            self.second = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.second is None:
            return self.first(x).squeeze(-1)
        hidden = torch.tanh(self.first(x))
        return self.second(hidden).squeeze(-1)

    def edge_contributions(self, x: torch.Tensor) -> torch.Tensor:
        if self.second is not None:
            return self.first.edge_contributions(x)
        return self.first.edge_contributions(x).squeeze(1)

    def parameter_count(self, *, active_only: bool = False) -> int:
        return sum(layer.parameter_count(active_only=active_only) for layer in self.modules() if isinstance(layer, KANLinear))


class BoundedResidualKAN(nn.Module):
    def __init__(self, input_dim: int, *, correction_bound: float, **kwargs: object) -> None:
        super().__init__()
        self.correction_bound = float(correction_bound)
        self.kan = SparseKANRegressor(input_dim, **kwargs)

    def forward(self, x: torch.Tensor, base_rul: torch.Tensor | None = None) -> torch.Tensor:
        raw = self.kan(x)
        correction = self.correction_bound * torch.tanh(raw)
        if base_rul is None:
            return correction
        return torch.clamp(base_rul + correction, min=0.0)

    def correction(self, x: torch.Tensor) -> torch.Tensor:
        return self.correction_bound * torch.tanh(self.kan(x))

    def parameter_count(self, *, active_only: bool = False) -> int:
        return self.kan.parameter_count(active_only=active_only)


class DirectKANRUL(nn.Module):
    def __init__(self, input_dim: int, *, rul_cap: float = 125.0, **kwargs: object) -> None:
        super().__init__()
        self.rul_cap = float(rul_cap)
        self.kan = SparseKANRegressor(input_dim, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rul_cap * torch.sigmoid(self.kan(x))

    def parameter_count(self, *, active_only: bool = False) -> int:
        return self.kan.parameter_count(active_only=active_only)
