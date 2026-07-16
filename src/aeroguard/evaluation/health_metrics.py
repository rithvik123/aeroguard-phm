"""Metrics for proxy health-index evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aeroguard.data.columns import CYCLE_COLUMN, UNIT_COLUMN


def health_index_metrics(
    frame: pd.DataFrame,
    health_column: str = "smoothed_health_index",
    rul_column: str = "true_rul_uncapped",
) -> dict[str, float | int | None]:
    """Evaluate health-index monotonicity and proxy-region separation."""
    if frame.empty:
        raise ValueError("Health-index metrics require a non-empty frame.")
    spearman = frame[[health_column, rul_column]].corr(method="spearman").iloc[0, 1]
    region_means = (
        frame.groupby("proxy_health_region")[health_column].mean().to_dict()
        if "proxy_health_region" in frame.columns
        else {}
    )
    healthy_mean = region_means.get("healthy_proxy")
    degradation_mean = region_means.get("degradation_proxy")
    critical_mean = region_means.get("critical_proxy")
    decreasing_fractions = []
    for _, group in frame.sort_values([UNIT_COLUMN, CYCLE_COLUMN]).groupby(UNIT_COLUMN):
        diffs = group[health_column].diff().dropna()
        if len(diffs):
            decreasing_fractions.append(float((diffs <= 0).mean()))
    separation = None
    if healthy_mean is not None and critical_mean is not None:
        separation = float(healthy_mean - critical_mean)
    return {
        "spearman_correlation_with_true_uncapped_rul": None
        if pd.isna(spearman)
        else float(spearman),
        "mean_health_index_healthy_proxy": None if healthy_mean is None else float(healthy_mean),
        "mean_health_index_degradation_proxy": None
        if degradation_mean is None
        else float(degradation_mean),
        "mean_health_index_critical_proxy": None if critical_mean is None else float(critical_mean),
        "per_engine_decreasing_trend_fraction": None
        if not decreasing_fractions
        else float(np.mean(decreasing_fractions)),
        "healthy_to_critical_separation": separation,
        "rows_evaluated": int(len(frame)),
        "engines_evaluated": int(frame[UNIT_COLUMN].nunique()),
    }
