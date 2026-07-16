"""Interpretability utilities for KAN edge functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from aeroguard.kan.kan_linear import KANLinear
from aeroguard.kan.pruning import collect_kan_layers


def edge_importance_frame(model: torch.nn.Module, feature_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(collect_kan_layers(model)):
        importance = layer.edge_importance().detach().cpu().numpy()
        for output_index in range(importance.shape[0]):
            for input_index in range(importance.shape[1]):
                feature = feature_names[input_index] if input_index < len(feature_names) else f"hidden_{input_index}"
                rows.append(
                    {
                        "layer_index": layer_index,
                        "output_index": output_index,
                        "input_index": input_index,
                        "feature_name": feature,
                        "edge_importance": float(importance[output_index, input_index]),
                        "active": bool(layer.edge_mask[output_index, input_index].item() > 0.0),
                    }
                )
    return pd.DataFrame(rows)


def univariate_curve_frame(layer: KANLinear, feature_names: list[str], *, points: int = 80) -> pd.DataFrame:
    x = torch.linspace(-layer.input_clamp, layer.input_clamp, points)
    rows = []
    with torch.no_grad():
        for input_index in range(layer.in_features):
            sample = torch.zeros(points, layer.in_features)
            sample[:, input_index] = x
            contributions = layer.edge_contributions(sample)[:, 0, input_index].cpu().numpy()
            feature = feature_names[input_index] if input_index < len(feature_names) else f"feature_{input_index}"
            for value, contribution in zip(x.cpu().numpy(), contributions):
                rows.append({"feature_name": feature, "normalized_value": float(value), "contribution": float(contribution)})
    return pd.DataFrame(rows)


def local_explanation(model: torch.nn.Module, x: np.ndarray, feature_names: list[str], *, top_k: int = 12) -> dict[str, Any]:
    tensor = torch.as_tensor(x.reshape(1, -1), dtype=torch.float32)
    with torch.no_grad():
        if hasattr(model, "kan"):
            contribution_tensor = model.kan.edge_contributions(tensor)
        else:
            contribution_tensor = model.edge_contributions(tensor)
    if contribution_tensor.ndim == 3:
        contribution_values = contribution_tensor[0, 0].detach().cpu().numpy()
    else:
        contribution_values = contribution_tensor[0].detach().cpu().numpy()
    order = np.argsort(np.abs(contribution_values))[::-1][:top_k]
    return {
        "top_contributions": [
            {
                "feature_name": feature_names[int(index)] if int(index) < len(feature_names) else f"feature_{int(index)}",
                "normalized_value": float(x[int(index)]),
                "contribution": float(contribution_values[int(index)]),
            }
            for index in order
        ],
        "contribution_sum": float(np.sum(contribution_values)),
    }
