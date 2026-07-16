"""Evaluation helpers for future Phase 5C physics-guided RUL runs."""

from __future__ import annotations

from typing import Any

import numpy as np

from aeroguard.deep.physics.violation_metrics import (
    cycle_rate_metrics,
    health_consistency_metrics,
    monotonicity_metrics,
    optimistic_error_metrics,
    regime_consistency_metrics,
    smoothness_metrics,
)


def trajectory_consistency_summary(
    earlier_prediction: object,
    later_prediction: object,
    cycle_gap: object,
    *,
    middle_prediction: object | None = None,
    tolerance: float = 0.0,
) -> dict[str, float]:
    summary: dict[str, float] = {}
    summary.update({f"monotonic_{key}": value for key, value in monotonicity_metrics(earlier_prediction, later_prediction, tolerance=tolerance).items()})
    summary.update({f"rate_{key}": value for key, value in cycle_rate_metrics(earlier_prediction, later_prediction, cycle_gap, tolerance=tolerance).items()})
    if middle_prediction is not None:
        summary.update({f"smooth_{key}": value for key, value in smoothness_metrics(earlier_prediction, middle_prediction, later_prediction, tolerance=tolerance).items()})
    return summary


def safety_summary(true_rul: object, predicted_rul: object, *, severe_threshold: float = 30.0, low_rul_threshold: float = 30.0) -> dict[str, float]:
    return optimistic_error_metrics(true_rul, predicted_rul, severe_threshold=severe_threshold, low_rul_threshold=low_rul_threshold)


def physics_summary_from_arrays(arrays: dict[str, Any], *, tolerance: float = 0.0) -> dict[str, float]:
    """Compose available standalone metrics from a dictionary of arrays."""

    summary: dict[str, float] = {}
    if {"earlier_prediction", "later_prediction", "cycle_gap"} <= set(arrays):
        summary.update(trajectory_consistency_summary(arrays["earlier_prediction"], arrays["later_prediction"], arrays["cycle_gap"], middle_prediction=arrays.get("middle_prediction"), tolerance=tolerance))
    if {"predicted_rul", "health_score"} <= set(arrays):
        summary.update({f"health_{key}": value for key, value in health_consistency_metrics(arrays["predicted_rul"], arrays["health_score"], arrays.get("earlier_index"), arrays.get("later_index"), tolerance=tolerance).items()})
    if {"latent_distance", "prediction_disagreement"} <= set(arrays):
        summary.update({f"regime_{key}": value for key, value in regime_consistency_metrics(arrays["latent_distance"], arrays["prediction_disagreement"], tolerance=tolerance).items()})
    if {"true_rul", "predicted_rul"} <= set(arrays):
        summary.update({f"safety_{key}": value for key, value in safety_summary(arrays["true_rul"], arrays["predicted_rul"]).items()})
    return summary


def normalized_robust_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted score used only with training-validation metrics in later selection."""

    total = 0.0
    for key, weight in weights.items():
        value = float(metrics.get(key, 0.0))
        if not np.isfinite(value) or not np.isfinite(float(weight)):
            raise ValueError("Robust score inputs must be finite.")
        total += float(weight) * value
    return float(total)
