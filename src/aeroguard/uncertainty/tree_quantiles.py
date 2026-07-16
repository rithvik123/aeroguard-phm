"""Random Forest tree-quantile interval helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def tree_prediction_matrix(model: object, x_values: object) -> np.ndarray:
    """Collect predictions from every tree in a fitted RandomForestRegressor."""
    estimators = getattr(model, "estimators_", None)
    if estimators is None or len(estimators) == 0:
        raise ValueError("Model does not expose fitted estimators_.")
    matrix = np.column_stack([tree.predict(x_values) for tree in estimators]).astype(float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Tree predictions must be a finite 2D array.")
    return matrix


def interval_quantiles_for_level(nominal_level: float) -> tuple[float, float]:
    level = float(nominal_level)
    if not 0.0 < level < 1.0:
        raise ValueError("Nominal level must be in (0, 1).")
    alpha = 1.0 - level
    return alpha / 2.0, 1.0 - alpha / 2.0


def tree_quantile_interval_frame(model: object, x_values: object, nominal_levels: Iterable[float], prefix: str = "") -> pd.DataFrame:
    matrix = tree_prediction_matrix(model, x_values)
    output: dict[str, np.ndarray] = {
        f"{prefix}tree_mean": matrix.mean(axis=1),
        f"{prefix}tree_std": matrix.std(axis=1, ddof=0),
        f"{prefix}tree_count": np.repeat(matrix.shape[1], matrix.shape[0]),
    }
    for level in nominal_levels:
        low_q, high_q = interval_quantiles_for_level(float(level))
        pct = int(round(float(level) * 100))
        lower = np.quantile(matrix, low_q, axis=1)
        upper = np.quantile(matrix, high_q, axis=1)
        output[f"{prefix}lower_{pct}"] = np.maximum(0.0, lower)
        output[f"{prefix}upper_{pct}"] = upper
        output[f"{prefix}raw_lower_{pct}"] = lower
    return pd.DataFrame(output)
