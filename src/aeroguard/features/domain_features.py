"""Domain-aware feature audit utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def domain_feature_audit(
    frame: pd.DataFrame,
    features: Iterable[str],
    reference_frame: pd.DataFrame,
    regime_column: str = "operating_regime",
) -> pd.DataFrame:
    """Audit retained features by subset and optional operating regime."""
    features = list(features)
    rows = []
    for feature in features:
        reference = pd.to_numeric(reference_frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        ref_mean = float(reference.mean()) if len(reference) else np.nan
        ref_std = float(reference.std(ddof=0)) if len(reference) else np.nan
        ref_p01 = float(reference.quantile(0.01)) if len(reference) else np.nan
        ref_p99 = float(reference.quantile(0.99)) if len(reference) else np.nan
        for keys, group in frame.groupby(["source_domain"] + ([regime_column] if regime_column in frame.columns else []), dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            values = pd.to_numeric(group[feature], errors="coerce")
            finite = values.replace([np.inf, -np.inf], np.nan).dropna()
            healthy = group[group["proxy_degradation_label"] == 0]
            degraded = group[group["proxy_degradation_label"] == 1]
            healthy_values = pd.to_numeric(healthy[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            degraded_values = pd.to_numeric(degraded[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            outside = None
            if len(finite) and np.isfinite(ref_p01) and np.isfinite(ref_p99):
                outside = float(((finite < ref_p01) | (finite > ref_p99)).mean())
            smd = None
            if len(finite) and np.isfinite(ref_std) and ref_std > 0:
                smd = float((finite.mean() - ref_mean) / ref_std)
            degradation_sensitivity = None
            if len(healthy_values) and len(degraded_values):
                scale = float(finite.std(ddof=0)) if len(finite) else 0.0
                if scale > 0:
                    degradation_sensitivity = float(abs(degraded_values.mean() - healthy_values.mean()) / scale)
            rows.append(
                {
                    "feature": feature,
                    "source_domain": keys[0],
                    "operating_regime": keys[1] if len(keys) > 1 else "all",
                    "mean": None if finite.empty else float(finite.mean()),
                    "std": None if finite.empty else float(finite.std(ddof=0)),
                    "median": None if finite.empty else float(finite.median()),
                    "iqr": None if finite.empty else float(finite.quantile(0.75) - finite.quantile(0.25)),
                    "healthy_region_mean": None if healthy_values.empty else float(healthy_values.mean()),
                    "degraded_region_mean": None if degraded_values.empty else float(degraded_values.mean()),
                    "missing_count": int(values.isna().sum()),
                    "infinite_count": int(np.isinf(values).sum()),
                    "outside_training_reference_range_fraction": outside,
                    "standardized_mean_difference": smd,
                    "regime_sensitivity": None,
                    "degradation_sensitivity": degradation_sensitivity,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    regime_span = (
        table.groupby("feature")["mean"]
        .agg(lambda series: float(pd.to_numeric(series, errors="coerce").max() - pd.to_numeric(series, errors="coerce").min()))
        .to_dict()
    )
    table["regime_sensitivity"] = table["feature"].map(regime_span)
    table["feature_category"] = table.apply(_feature_category, axis=1)
    return table


def _feature_category(row: pd.Series) -> str:
    regime = row.get("regime_sensitivity")
    degradation = row.get("degradation_sensitivity")
    smd = row.get("standardized_mean_difference")
    regime_high = regime is not None and np.isfinite(regime) and float(regime) > 1.0
    degradation_high = degradation is not None and np.isfinite(degradation) and float(degradation) > 0.5
    unstable = smd is not None and np.isfinite(smd) and abs(float(smd)) > 1.0
    if regime_high and degradation_high:
        return "both_condition_and_degradation_sensitive"
    if regime_high:
        return "primarily_operating_condition_driven"
    if degradation_high:
        return "primarily_degradation_driven"
    if unstable:
        return "unstable_across_domains"
    return "stable_across_domains"
