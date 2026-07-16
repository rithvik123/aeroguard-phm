"""Metrics for RUL prediction intervals."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from aeroguard.evaluation.metrics import regression_metrics


def _arrays(true: Iterable[float], pred: Iterable[float], lower: Iterable[float], upper: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(list(true), dtype=float)
    p = np.asarray(list(pred), dtype=float)
    lo = np.asarray(list(lower), dtype=float)
    hi = np.asarray(list(upper), dtype=float)
    if len(y) == 0 or not (len(y) == len(p) == len(lo) == len(hi)):
        raise ValueError("Metric arrays must be non-empty and aligned.")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("Metric arrays must contain only finite values.")
    return y, p, lo, hi


def interval_metrics(true: Iterable[float], pred: Iterable[float], lower: Iterable[float], upper: Iterable[float], nominal_level: float) -> dict[str, float | int]:
    level = float(nominal_level)
    if not 0.0 < level < 1.0:
        raise ValueError("nominal_level must be in (0, 1).")
    y, p, lo, hi = _arrays(true, pred, lower, upper)
    crossing = lo > hi
    ordered_lo = np.minimum(lo, hi)
    ordered_hi = np.maximum(lo, hi)
    covered = (y >= ordered_lo) & (y <= ordered_hi)
    width = ordered_hi - ordered_lo
    alpha = 1.0 - level
    penalties = (2.0 / alpha) * ((ordered_lo - y) * (y < ordered_lo) + (y - ordered_hi) * (y > ordered_hi))
    score = width + penalties
    denom = np.maximum(np.abs(y), 1.0)
    coverage = float(covered.mean())
    coverage_error = coverage - level
    return {
        "nominal_level": level,
        "engine_count": int(len(y)),
        "coverage": coverage,
        "coverage_error": coverage_error,
        "absolute_coverage_error": abs(coverage_error),
        "undercoverage_amount": max(0.0, level - coverage),
        "overcoverage_amount": max(0.0, coverage - level),
        "mean_interval_width": float(width.mean()),
        "median_interval_width": float(np.median(width)),
        "mean_normalized_interval_width": float(np.mean(width / denom)),
        "winkler_interval_score": float(score.mean()),
        "mean_interval_score": float(score.mean()),
        "lower_bound_violation_rate": float((y < ordered_lo).mean()),
        "upper_bound_violation_rate": float((y > ordered_hi).mean()),
        "interval_crossing_count": int(crossing.sum()),
        "interval_crossing_rate": float(crossing.mean()),
    }


def point_metrics(true: Iterable[float], pred: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(true), dtype=float)
    p = np.asarray(list(pred), dtype=float)
    metrics = regression_metrics(y, p)
    residual = p - y
    metrics.update(
        {
            "mean_signed_error": float(residual.mean()),
            "overly_optimistic_fraction": float((residual > 0).mean()),
            "conservative_fraction": float((residual < 0).mean()),
        }
    )
    return metrics
