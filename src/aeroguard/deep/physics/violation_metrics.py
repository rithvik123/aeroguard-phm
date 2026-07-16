"""Standalone physics-violation metrics for later Phase 5C evaluation."""

from __future__ import annotations

import numpy as np


def _array(values: object, name: str, *, allow_empty: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        if allow_empty:
            return arr
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite.")
    return arr


def monotonicity_metrics(earlier_prediction: object, later_prediction: object, *, tolerance: float = 0.0) -> dict[str, float]:
    earlier = _array(earlier_prediction, "earlier_prediction")
    later = _array(later_prediction, "later_prediction")
    _same_shape(earlier, later)
    violation = np.maximum(later - earlier - float(tolerance), 0.0)
    count = int((violation > 0).sum())
    total = int(violation.size)
    return {
        "violation_count": float(count),
        "violation_rate": float(count / total) if total else 0.0,
        "mean_violation_magnitude": float(violation.mean()) if total else 0.0,
        "maximum_violation_magnitude": float(violation.max()) if total else 0.0,
    }


def cycle_rate_metrics(earlier_prediction: object, later_prediction: object, cycle_gap: object, *, tolerance: float = 0.0) -> dict[str, float]:
    earlier = _array(earlier_prediction, "earlier_prediction")
    later = _array(later_prediction, "later_prediction")
    gap = _array(cycle_gap, "cycle_gap")
    _same_shape(earlier, later, gap)
    if (gap <= 0).any():
        raise ValueError("cycle_gap values must be positive.")
    residual = later - earlier + gap
    abs_residual = np.abs(residual)
    return {
        "mean_rate_residual": float(residual.mean()) if residual.size else 0.0,
        "median_rate_residual": float(np.median(residual)) if residual.size else 0.0,
        "rate_rmse": float(np.sqrt(np.mean(np.square(residual)))) if residual.size else 0.0,
        "rate_violation_rate": float((abs_residual > float(tolerance)).mean()) if residual.size else 0.0,
    }


def smoothness_metrics(earlier_prediction: object, middle_prediction: object, later_prediction: object, *, tolerance: float = 0.0) -> dict[str, float]:
    earlier = _array(earlier_prediction, "earlier_prediction")
    middle = _array(middle_prediction, "middle_prediction")
    later = _array(later_prediction, "later_prediction")
    _same_shape(earlier, middle, later)
    second = np.abs(later - (2.0 * middle) + earlier)
    return {
        "mean_absolute_second_difference": float(second.mean()) if second.size else 0.0,
        "median_absolute_second_difference": float(np.median(second)) if second.size else 0.0,
        "p95_second_difference": float(np.quantile(second, 0.95)) if second.size else 0.0,
        "smoothness_violation_rate": float((second > float(tolerance)).mean()) if second.size else 0.0,
    }


def health_consistency_metrics(
    predicted_rul: object,
    health_score: object,
    earlier_index: object | None = None,
    later_index: object | None = None,
    *,
    tolerance: float = 0.0,
) -> dict[str, float]:
    rul = _array(predicted_rul, "predicted_rul")
    health = _array(health_score, "health_score")
    _same_shape(rul, health)
    if rul.size == 0:
        return {"health_rul_spearman": 0.0, "health_monotonicity_violation_rate": 0.0, "pairwise_health_direction_agreement": 0.0}
    corr = _spearman(rul, health)
    violation_rate = 0.0
    agreement = 0.0
    if earlier_index is not None and later_index is not None:
        earlier = np.asarray(earlier_index, dtype=int).reshape(-1)
        later = np.asarray(later_index, dtype=int).reshape(-1)
        _same_shape(earlier, later)
        if len(earlier):
            health_violation = health[later] - health[earlier] > float(tolerance)
            rul_direction = rul[earlier] >= rul[later] - float(tolerance)
            health_direction = health[earlier] >= health[later] - float(tolerance)
            violation_rate = float(health_violation.mean())
            agreement = float((rul_direction == health_direction).mean())
    return {
        "health_rul_spearman": corr,
        "health_monotonicity_violation_rate": violation_rate,
        "pairwise_health_direction_agreement": agreement,
    }


def regime_consistency_metrics(latent_distance: object, prediction_disagreement: object, *, tolerance: float = 0.0) -> dict[str, float]:
    latent = _array(latent_distance, "latent_distance")
    disagreement = _array(prediction_disagreement, "prediction_disagreement")
    _same_shape(latent, disagreement)
    return {
        "mean_latent_distance": float(latent.mean()) if latent.size else 0.0,
        "mean_prediction_disagreement": float(disagreement.mean()) if disagreement.size else 0.0,
        "regime_consistency_violation_rate": float((disagreement > float(tolerance)).mean()) if disagreement.size else 0.0,
    }


def optimistic_error_metrics(
    true_rul: object,
    predicted_rul: object,
    *,
    severe_threshold: float = 30.0,
    low_rul_threshold: float = 30.0,
) -> dict[str, float]:
    true = _array(true_rul, "true_rul")
    pred = _array(predicted_rul, "predicted_rul")
    _same_shape(true, pred)
    optimistic = np.maximum(pred - true, 0.0)
    positive = optimistic > 0.0
    severe = optimistic > float(severe_threshold)
    low = true <= float(low_rul_threshold)
    return {
        "optimistic_prediction_rate": float(positive.mean()) if optimistic.size else 0.0,
        "severe_optimistic_prediction_rate": float(severe.mean()) if optimistic.size else 0.0,
        "mean_optimistic_error_magnitude": float(optimistic[positive].mean()) if positive.any() else 0.0,
        "maximum_optimistic_error_magnitude": float(optimistic.max()) if optimistic.size else 0.0,
        "low_rul_optimistic_error_rate": float(positive[low].mean()) if low.any() else 0.0,
    }


def _same_shape(*arrays: np.ndarray) -> None:
    if len({arr.shape for arr in arrays}) != 1:
        raise ValueError("Metric inputs must be aligned.")


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _rank(left)
    right_rank = _rank(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = float(np.mean(np.arange(start, stop)))
        start = stop
    return ranks
