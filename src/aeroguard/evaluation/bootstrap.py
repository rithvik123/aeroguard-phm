"""Engine-level bootstrap confidence intervals."""

from __future__ import annotations

from typing import Callable, Any

import numpy as np
import pandas as pd

from aeroguard.data.columns import UNIT_COLUMN


MetricFunction = Callable[[pd.DataFrame], float | int | None]


def percentile_interval(values: list[float], confidence_level: float) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    alpha = 1.0 - float(confidence_level)
    low = float(np.percentile(values, 100 * alpha / 2))
    high = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return low, high


def bootstrap_engine_metrics(
    engine_frame: pd.DataFrame,
    metric_functions: dict[str, MetricFunction],
    n_samples: int,
    confidence_level: float,
    seed: int,
    engine_column: str = UNIT_COLUMN,
) -> dict[str, dict[str, float | int | None]]:
    """Bootstrap one-row-per-engine metrics by resampling engines."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1).")
    if engine_frame.empty:
        return {
            name: {
                "estimate": None,
                "ci_lower": None,
                "ci_upper": None,
                "valid_replicates": 0,
                "bootstrap_samples": int(n_samples),
                "confidence_level": float(confidence_level),
            }
            for name in metric_functions
        }
    if engine_column not in engine_frame.columns:
        raise ValueError(f"Missing engine column: {engine_column}")

    rng = np.random.default_rng(int(seed))
    engines = np.array(sorted(engine_frame[engine_column].unique()), dtype=int)
    grouped = {int(engine): group.copy() for engine, group in engine_frame.groupby(engine_column)}
    replicates: dict[str, list[float]] = {name: [] for name in metric_functions}
    estimates: dict[str, float | None] = {}
    for name, func in metric_functions.items():
        value = func(engine_frame)
        estimates[name] = None if value is None or not np.isfinite(float(value)) else float(value)

    for _ in range(int(n_samples)):
        sampled = rng.choice(engines, size=len(engines), replace=True)
        pieces = []
        for draw_index, engine in enumerate(sampled):
            piece = grouped[int(engine)].copy()
            piece[engine_column] = draw_index + 1
            pieces.append(piece)
        sample_frame = pd.concat(pieces, ignore_index=True)
        for name, func in metric_functions.items():
            value = func(sample_frame)
            if value is not None and np.isfinite(float(value)):
                replicates[name].append(float(value))

    result: dict[str, dict[str, float | int | None]] = {}
    for name, values in replicates.items():
        low, high = percentile_interval(values, confidence_level)
        result[name] = {
            "estimate": estimates[name],
            "ci_lower": low,
            "ci_upper": high,
            "valid_replicates": int(len(values)),
            "bootstrap_samples": int(n_samples),
            "confidence_level": float(confidence_level),
        }
    return result


def alert_engine_metric_functions() -> dict[str, MetricFunction]:
    """Metric functions for engine summary rows emitted by onset evaluation."""

    def fraction(column: str) -> MetricFunction:
        return lambda frame: float(frame[column].astype(bool).mean()) if len(frame) else None

    def false_alarm_rate(frame: pd.DataFrame) -> float | None:
        return float(frame["false_alarm"].astype(bool).mean()) if len(frame) else None

    def valid_median(column: str) -> MetricFunction:
        def metric(frame: pd.DataFrame) -> float | None:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            return None if values.empty else float(values.median())

        return metric

    def before(column: str) -> MetricFunction:
        def metric(frame: pd.DataFrame) -> float | None:
            detected = frame[frame["detected"].astype(bool)]
            if detected.empty:
                return None
            return float(sum(value is True for value in detected[column].tolist()) / len(detected))

        return metric

    return {
        "detection_rate": fraction("detected"),
        "missed_engine_rate": fraction("missed"),
        "false_alarm_engine_rate": false_alarm_rate,
        "median_lead_time": valid_median("lead_time_before_failure"),
        "median_detection_delay": valid_median("detection_delay"),
        "detected_before_60_fraction": before("before_60_cycles_rul"),
        "detected_before_30_fraction": before("before_30_cycles_rul"),
        "critical_region_recall": before("before_critical_threshold"),
        "median_alert_transitions": valid_median("alert_transition_count"),
    }
