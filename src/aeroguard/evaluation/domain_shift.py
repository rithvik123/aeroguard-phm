"""Descriptive domain-shift diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from aeroguard.data.columns import CYCLE_COLUMN, UNIT_COLUMN


def finite_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).dropna()


def population_stability_index(
    reference: pd.Series,
    comparison: pd.Series,
    bins: int = 10,
    epsilon: float = 1.0e-6,
) -> float | None:
    """Compute lightweight PSI using bins derived from the reference distribution."""
    ref = finite_series(reference)
    comp = finite_series(comparison)
    if ref.empty or comp.empty:
        return None
    quantiles = np.linspace(0, 1, int(bins) + 1)
    edges = np.unique(np.quantile(ref.to_numpy(dtype=float), quantiles))
    if len(edges) < 2:
        return 0.0 if np.isclose(ref.iloc[0], comp.iloc[0]) else None
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    comp_counts, _ = np.histogram(comp, bins=edges)
    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), epsilon)
    comp_pct = np.maximum(comp_counts / max(comp_counts.sum(), 1), epsilon)
    return float(np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct)))


def feature_shift_table(
    reference_healthy: pd.DataFrame,
    fd001_test: pd.DataFrame,
    fd003_test: pd.DataFrame,
    features: list[str],
    psi_bins: int = 10,
) -> pd.DataFrame:
    """Compute per-feature FD001 healthy reference versus FD001/FD003 test shift."""
    rows: list[dict[str, Any]] = []
    for feature in features:
        ref = finite_series(reference_healthy[feature])
        fd001 = finite_series(fd001_test[feature])
        fd003 = finite_series(fd003_test[feature])
        ref_mean = float(ref.mean()) if not ref.empty else None
        ref_std = float(ref.std(ddof=0)) if len(ref) else None
        ref_p01 = float(ref.quantile(0.01)) if not ref.empty else None
        ref_p99 = float(ref.quantile(0.99)) if not ref.empty else None

        def out_fraction(series: pd.Series) -> float | None:
            if series.empty or ref_p01 is None or ref_p99 is None:
                return None
            return float(((series < ref_p01) | (series > ref_p99)).mean())

        def smd(series: pd.Series) -> float | None:
            if series.empty or ref_mean is None or not ref_std or ref_std == 0:
                return None
            return float((series.mean() - ref_mean) / ref_std)

        def missing_count(frame: pd.DataFrame) -> int:
            return int(pd.to_numeric(frame[feature], errors="coerce").isna().sum())

        def infinite_count(frame: pd.DataFrame) -> int:
            numeric = pd.to_numeric(frame[feature], errors="coerce")
            return int(np.isinf(numeric).sum())

        rows.append(
            {
                "feature": feature,
                "fd001_healthy_train_mean": ref_mean,
                "fd001_healthy_train_std": ref_std,
                "fd001_test_mean": None if fd001.empty else float(fd001.mean()),
                "fd003_test_mean": None if fd003.empty else float(fd003.mean()),
                "fd001_test_standardized_mean_difference": smd(fd001),
                "fd003_standardized_mean_difference": smd(fd003),
                "fd001_healthy_p01": ref_p01,
                "fd001_healthy_p99": ref_p99,
                "fd001_test_outside_healthy_1_99_fraction": out_fraction(fd001),
                "fd003_outside_healthy_1_99_fraction": out_fraction(fd003),
                "fd001_healthy_median": None if ref.empty else float(ref.median()),
                "fd001_test_median": None if fd001.empty else float(fd001.median()),
                "fd003_median": None if fd003.empty else float(fd003.median()),
                "fd001_healthy_iqr": None if ref.empty else float(ref.quantile(0.75) - ref.quantile(0.25)),
                "fd001_test_iqr": None if fd001.empty else float(fd001.quantile(0.75) - fd001.quantile(0.25)),
                "fd003_iqr": None if fd003.empty else float(fd003.quantile(0.75) - fd003.quantile(0.25)),
                "fd001_healthy_missing_count": missing_count(reference_healthy),
                "fd001_test_missing_count": missing_count(fd001_test),
                "fd003_missing_count": missing_count(fd003_test),
                "fd001_healthy_infinite_count": infinite_count(reference_healthy),
                "fd001_test_infinite_count": infinite_count(fd001_test),
                "fd003_infinite_count": infinite_count(fd003_test),
                "fd001_test_psi": population_stability_index(reference_healthy[feature], fd001_test[feature], bins=psi_bins),
                "fd003_psi": population_stability_index(reference_healthy[feature], fd003_test[feature], bins=psi_bins),
            }
        )
    return pd.DataFrame(rows)


def trajectory_summary(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    grouped = frame.sort_values([UNIT_COLUMN, CYCLE_COLUMN]).groupby(UNIT_COLUMN)
    lengths = grouped.size()
    first = grouped.head(1)
    last = grouped.tail(1)
    return {
        f"{label}_engine_count": int(frame[UNIT_COLUMN].nunique()),
        f"{label}_row_count": int(len(frame)),
        f"{label}_trajectory_length_median": float(lengths.median()) if len(lengths) else None,
        f"{label}_trajectory_length_iqr": float(lengths.quantile(0.75) - lengths.quantile(0.25)) if len(lengths) else None,
        f"{label}_initial_true_rul_median": float(first["true_rul_uncapped"].median()) if "true_rul_uncapped" in first else None,
        f"{label}_final_true_rul_median": float(last["true_rul_uncapped"].median()) if "true_rul_uncapped" in last else None,
        f"{label}_begins_degraded_fraction": float(first["proxy_degradation_label"].astype(bool).mean())
        if "proxy_degradation_label" in first
        else None,
    }


def distribution_summary(frame: pd.DataFrame, columns: list[str], label: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        values = finite_series(frame[column])
        summary[f"{label}_{column}_mean"] = None if values.empty else float(values.mean())
        summary[f"{label}_{column}_median"] = None if values.empty else float(values.median())
        summary[f"{label}_{column}_iqr"] = None if values.empty else float(values.quantile(0.75) - values.quantile(0.25))
    return summary
