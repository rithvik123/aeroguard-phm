"""Regression metrics and NASA asymmetric RUL scoring."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _as_finite_arrays(y_true: object, y_pred: object) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("Metric inputs must be one-dimensional.")
    if len(true) == 0:
        raise ValueError("Metric inputs must not be empty.")
    if len(true) != len(pred):
        raise ValueError("Metric input lengths must match.")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Metric inputs must contain only finite values.")
    return true, pred


def mean_absolute_error(y_true: object, y_pred: object) -> float:
    """Mean absolute RUL error in cycles."""
    true, pred = _as_finite_arrays(y_true, y_pred)
    return float(np.mean(np.abs(pred - true)))


def root_mean_squared_error(y_true: object, y_pred: object) -> float:
    """Root mean squared RUL error in cycles."""
    true, pred = _as_finite_arrays(y_true, y_pred)
    return float(math.sqrt(np.mean(np.square(pred - true))))


def r2_score(y_true: object, y_pred: object) -> float:
    """Coefficient of determination."""
    true, pred = _as_finite_arrays(y_true, y_pred)
    denominator = float(np.sum(np.square(true - np.mean(true))))
    if denominator == 0.0:
        return float("nan")
    numerator = float(np.sum(np.square(true - pred)))
    return float(1.0 - numerator / denominator)


def nasa_asymmetric_score(y_true: object, y_pred: object) -> float:
    """NASA PHM-style asymmetric score.

    The residual convention is ``predicted_RUL - true_RUL``.

    * residual < 0 means the model predicted less remaining life than actual.
      This is conservative, often causing earlier maintenance.
    * residual > 0 means the model predicted more remaining life than actual.
      This is late or overly optimistic and receives the stronger penalty.

    For each engine residual ``d``:
    ``exp(-d / 13) - 1`` is used when ``d < 0`` and
    ``exp(d / 10) - 1`` is used when ``d >= 0``.
    Lower scores are better; perfect predictions score 0.
    """
    true, pred = _as_finite_arrays(y_true, y_pred)
    residual = pred - true
    penalties = np.where(
        residual < 0,
        np.exp(-residual / 13.0) - 1.0,
        np.exp(residual / 10.0) - 1.0,
    )
    return float(np.sum(penalties))


def regression_metrics(y_true: object, y_pred: object) -> dict[str, float]:
    """Compute all required RUL regression metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "nasa_score": nasa_asymmetric_score(y_true, y_pred),
    }


def prediction_direction(residual: float) -> str:
    """Label prediction direction using residual = predicted - true."""
    if residual < 0:
        return "under_prediction_conservative"
    if residual > 0:
        return "over_prediction_optimistic"
    return "perfect"


def per_engine_prediction_frame(
    unit_ids: object,
    y_true: object,
    y_pred: object,
    model_name: str,
) -> pd.DataFrame:
    """Create per-engine prediction detail rows for final test evaluation."""
    true, pred = _as_finite_arrays(y_true, y_pred)
    unit_array = np.asarray(unit_ids)
    if len(unit_array) != len(true):
        raise ValueError("unit_ids length must match y_true and y_pred.")
    residual = pred - true
    return pd.DataFrame(
        {
            "model": model_name,
            "unit_id": unit_array.astype(int),
            "true_rul": true,
            "predicted_rul": pred,
            "residual": residual,
            "absolute_error": np.abs(residual),
            "squared_error": np.square(residual),
            "prediction_direction": [prediction_direction(value) for value in residual],
        }
    )
