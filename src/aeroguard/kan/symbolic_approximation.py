"""Simple symbolic curve approximations for stable KAN functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _basis(name: str, x: np.ndarray) -> np.ndarray:
    if name == "linear":
        return np.column_stack([np.ones_like(x), x])
    if name == "quadratic":
        return np.column_stack([np.ones_like(x), x, x**2])
    if name == "cubic":
        return np.column_stack([np.ones_like(x), x, x**2, x**3])
    if name == "tanh":
        return np.column_stack([np.ones_like(x), np.tanh(x)])
    if name == "sigmoid":
        return np.column_stack([np.ones_like(x), 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))])
    if name == "softplus":
        return np.column_stack([np.ones_like(x), np.log1p(np.exp(np.clip(x, -20, 20)))])
    if name == "log1p_abs":
        return np.column_stack([np.ones_like(x), np.log1p(np.abs(x))])
    if name == "safe_reciprocal":
        return np.column_stack([np.ones_like(x), 1.0 / (1.0 + np.abs(x))])
    if name == "safe_exp":
        return np.column_stack([np.ones_like(x), np.exp(np.clip(x, -4, 4))])
    raise ValueError(name)


def approximate_curve(x: np.ndarray, y: np.ndarray, *, fidelity_rmse: float = 0.05) -> dict[str, object]:
    best: dict[str, object] | None = None
    for name in ["linear", "quadratic", "cubic", "tanh", "sigmoid", "softplus", "log1p_abs", "safe_reciprocal", "safe_exp"]:
        matrix = _basis(name, x)
        coef, *_ = np.linalg.lstsq(matrix, y, rcond=None)
        pred = matrix @ coef
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        max_dev = float(np.max(np.abs(pred - y)))
        row = {"function": name, "coefficients": [float(v) for v in coef], "approximation_rmse": rmse, "maximum_deviation": max_dev, "accepted": bool(rmse <= fidelity_rmse)}
        if best is None or rmse < float(best["approximation_rmse"]):
            best = row
    assert best is not None
    return best


def approximate_curves(curves: pd.DataFrame, *, fidelity_rmse: float = 0.05) -> pd.DataFrame:
    rows = []
    for feature_name, group in curves.groupby("feature_name", observed=False):
        result = approximate_curve(group["normalized_value"].to_numpy(dtype=float), group["contribution"].to_numpy(dtype=float), fidelity_rmse=fidelity_rmse)
        rows.append({"feature_name": feature_name, **result})
    return pd.DataFrame(rows)
