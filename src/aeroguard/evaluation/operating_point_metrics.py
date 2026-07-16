"""Operating-point scoring and utility metrics."""

from __future__ import annotations

from typing import Any


def fraction(numerator: float | int | None, denominator: float | int | None) -> float:
    if numerator is None or denominator in {None, 0}:
        return 0.0
    return float(numerator) / float(denominator)


def compute_operating_utility(
    row_metrics: dict[str, Any],
    engine_metrics: dict[str, Any],
    transition_count: int,
    weights: dict[str, float],
) -> float:
    """Transparent weighted validation utility."""
    engines = engine_metrics.get("engines_evaluated", 0)
    detected = engine_metrics.get("detected_engines", 0)
    missed = engine_metrics.get("missed_engines", 0)
    false_alarm = engine_metrics.get("false_alarm_engine_count", 0)
    before_30 = engine_metrics.get("detections_before_30_cycles_rul", 0)
    detection_rate = engine_metrics.get("detection_rate") or fraction(detected, engines)
    before_30_fraction = fraction(before_30, detected)
    missed_fraction = fraction(missed, engines)
    false_alarm_fraction = fraction(false_alarm, engines)
    healthy_fpr = row_metrics.get("false_positive_rate") or 0.0
    transition_norm = fraction(transition_count, max(engines, 1) * 10)
    late_after_critical = 1.0 - before_30_fraction
    utility = (
        weights.get("detection_reward", 0.0) * detection_rate
        + weights.get("early_warning_reward", 0.0) * before_30_fraction
        - weights.get("missed_engine_penalty", 0.0) * missed_fraction
        - weights.get("false_alarm_engine_penalty", 0.0) * false_alarm_fraction
        - weights.get("healthy_fpr_penalty", 0.0) * healthy_fpr
        - weights.get("instability_penalty", 0.0) * transition_norm
        - weights.get("late_after_critical_penalty", 0.0) * late_after_critical
    )
    return float(utility)


def rank_operating_points(rows: list[dict[str, Any]], profile_name: str) -> list[dict[str, Any]]:
    """Return rows sorted by utility descending for a named profile."""
    utility_column = f"utility_{profile_name}"
    return sorted(rows, key=lambda row: row.get(utility_column, float("-inf")), reverse=True)
