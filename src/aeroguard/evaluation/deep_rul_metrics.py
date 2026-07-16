"""Deep RUL point-prediction metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aeroguard.evaluation.metrics import nasa_asymmetric_score, regression_metrics


def prediction_direction(residual: float) -> str:
    if residual > 0:
        return "over_prediction_optimistic"
    if residual < 0:
        return "under_prediction_conservative"
    return "perfect"


def deep_point_metrics(true: object, predicted: object, severe_optimistic_threshold: float = 30.0) -> dict[str, float]:
    y = np.asarray(true, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metric inputs must be non-empty, aligned, and finite.")
    residual = p - y
    metrics = regression_metrics(y, p)
    metrics.update(
        {
            "nasa_score": nasa_asymmetric_score(y, p),
            "mean_signed_error": float(residual.mean()),
            "median_absolute_error": float(np.median(np.abs(residual))),
            "p90_absolute_error": float(np.quantile(np.abs(residual), 0.90)),
            "optimistic_prediction_rate": float((residual > 0).mean()),
            "conservative_prediction_rate": float((residual < 0).mean()),
            "severe_optimistic_error_rate": float((residual > float(severe_optimistic_threshold)).mean()),
        }
    )
    return metrics


def metrics_by_group(frame: pd.DataFrame, group_column: str, severe_optimistic_threshold: float = 30.0) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(group_column, dropna=False, observed=False):
        rows.append({group_column: value, "engine_count": int(len(group)), **deep_point_metrics(group["true_rul"], group["predicted_rul"], severe_optimistic_threshold)})
    return pd.DataFrame(rows)
