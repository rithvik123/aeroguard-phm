"""Edge pruning helpers for KAN models."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aeroguard.kan.kan_linear import KANLinear


@dataclass(frozen=True)
class PruningReport:
    edges_before: int
    edges_after: int
    threshold: float


def prune_layer_by_quantile(layer: KANLinear, quantile: float) -> PruningReport:
    importance = layer.edge_importance()
    edges_before = int(layer.edge_mask.sum().item())
    if edges_before == 0:
        return PruningReport(0, 0, float("inf"))
    threshold = float(torch.quantile(importance.reshape(-1), float(quantile)).item())
    mask = (importance >= threshold).to(dtype=layer.edge_mask.dtype, device=layer.edge_mask.device)
    if int(mask.sum().item()) == 0:
        flat_index = int(torch.argmax(importance.reshape(-1)).item())
        mask.reshape(-1)[flat_index] = 1.0
    layer.apply_pruning_mask(mask)
    return PruningReport(edges_before=edges_before, edges_after=int(mask.sum().item()), threshold=threshold)


def collect_kan_layers(module: torch.nn.Module) -> list[KANLinear]:
    return [layer for layer in module.modules() if isinstance(layer, KANLinear)]
