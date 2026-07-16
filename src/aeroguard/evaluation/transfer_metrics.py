"""Transfer-evaluation utilities for multidomain PHM."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def method_utility(row: pd.Series | dict[str, Any], weights: dict[str, float]) -> float:
    """Robust multidomain utility for method selection."""
    return float(
        safe_float(weights.get("detection_rate")) * safe_float(row.get("engine_detection_rate_mean", row.get("engine_detection_rate")))
        + safe_float(weights.get("critical_region_recall")) * safe_float(row.get("critical_region_recall_mean", row.get("critical_region_recall")))
        + safe_float(weights.get("detected_before_30_fraction")) * safe_float(row.get("detected_before_30_fraction_mean", row.get("detected_before_30_fraction")))
        + safe_float(weights.get("detected_before_60_fraction")) * safe_float(row.get("detected_before_60_fraction_mean", row.get("detected_before_60_fraction")))
        - safe_float(weights.get("false_alarm_engine_rate")) * safe_float(row.get("false_alarm_engine_rate_mean", row.get("false_alarm_engine_rate")))
        - safe_float(weights.get("max_domain_false_alarm_engine_rate")) * safe_float(row.get("false_alarm_engine_rate_max", 0.0))
        - safe_float(weights.get("missed_engine_rate")) * safe_float(row.get("missed_engine_rate_mean", row.get("missed_engine_rate")))
        - safe_float(weights.get("healthy_region_false_positive_rate")) * safe_float(row.get("healthy_region_false_positive_rate_mean", row.get("healthy_region_false_positive_rate")))
        - safe_float(weights.get("fold_variability")) * safe_float(row.get("utility_std", 0.0))
        - safe_float(weights.get("domain_variability")) * safe_float(row.get("domain_detection_rate_std", 0.0))
        - safe_float(weights.get("alert_instability")) * safe_float(row.get("mean_alert_transitions_mean", row.get("mean_alert_transitions", 0.0)))
    )


def summarize_method_metrics(
    metrics: pd.DataFrame,
    weights: dict[str, float],
    feasibility_constraints: dict[str, float],
) -> pd.DataFrame:
    """Aggregate validation rows by method and compute robust utility."""
    rows = []
    for method_id, group in metrics.groupby("method_id"):
        row: dict[str, Any] = {"method_id": method_id, "evaluation_count": int(len(group))}
        for metric in [
            "engine_detection_rate",
            "missed_engine_rate",
            "false_alarm_engine_rate",
            "healthy_region_false_positive_rate",
            "critical_region_recall",
            "median_lead_time",
            "median_detection_delay",
            "detected_before_30_fraction",
            "detected_before_60_fraction",
            "mean_alert_transitions",
        ]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna() if metric in group.columns else pd.Series(dtype=float)
            row[f"{metric}_mean"] = None if values.empty else float(values.mean())
            row[f"{metric}_std"] = None if values.empty else float(values.std(ddof=0))
            row[f"{metric}_max"] = None if values.empty else float(values.max())
            row[f"{metric}_min"] = None if values.empty else float(values.min())
        utilities = group.apply(lambda item: method_utility(item, weights), axis=1)
        row["mean_utility"] = float(utilities.mean())
        row["utility_std"] = float(utilities.std(ddof=0))
        row["domain_detection_rate_std"] = float(
            group.groupby("validation_domain")["engine_detection_rate"].mean().std(ddof=0)
        ) if "validation_domain" in group.columns else 0.0
        row["robust_utility"] = method_utility(row, weights)
        failed = failed_constraints(row, feasibility_constraints)
        row["feasible"] = len(failed) == 0
        row["failed_constraints"] = "; ".join(failed)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["feasible", "robust_utility", "mean_utility", "false_alarm_engine_rate_mean", "method_id"],
        ascending=[False, False, False, True, True],
    )


def failed_constraints(row: dict[str, Any], constraints: dict[str, float]) -> list[str]:
    checks = {
        "max_mean_false_alarm_engine_rate": ("false_alarm_engine_rate_mean", lambda value, limit: value <= limit),
        "max_mean_healthy_region_false_positive_rate": ("healthy_region_false_positive_rate_mean", lambda value, limit: value <= limit),
        "minimum_mean_detection_rate": ("engine_detection_rate_mean", lambda value, limit: value >= limit),
        "minimum_mean_critical_region_recall": ("critical_region_recall_mean", lambda value, limit: value >= limit),
    }
    failed = []
    for key, limit in constraints.items():
        if key not in checks:
            raise ValueError(f"Unsupported feasibility constraint: {key}")
        metric, predicate = checks[key]
        value = row.get(metric)
        if value is None or not predicate(float(value), float(limit)):
            failed.append(f"{key} failed: {metric}={value}, limit={limit}")
    return failed


def classify_transfer(criteria: dict[str, dict[str, float]], fd004_metrics: dict[str, Any], rul_metrics: dict[str, Any]) -> dict[str, Any]:
    values = {
        "fd004_detection_rate": safe_float(fd004_metrics["engine_level"].get("detection_rate")),
        "fd004_false_alarm_engine_rate": safe_float(fd004_metrics["engine_level"].get("false_alarm_engine_rate")),
        "fd004_missed_engine_rate": safe_float(fd004_metrics["engine_level"].get("missed_engine_rate")),
        "fd004_critical_region_recall": safe_float(fd004_metrics["row_level"].get("critical_region_recall")),
        "fd004_rul_mae": safe_float(rul_metrics.get("mae"), default=float("inf")),
    }
    for label in ["strong", "moderate", "weak"]:
        failed = []
        for key, limit in criteria.get(label, {}).items():
            value = values.get(key)
            if value is None:
                failed.append(f"{key}=missing")
            elif "false_alarm" in key or "missed" in key or key.endswith("_mae"):
                if value > float(limit):
                    failed.append(f"{key}={value} > {limit}")
            elif value < float(limit):
                failed.append(f"{key}={value} < {limit}")
        if not failed:
            return {"classification": f"{label.title()} generalization", "criteria_values": values, "failed_criteria": failed}
    return {
        "classification": "Failed external transfer",
        "criteria_values": values,
        "failed_criteria": ["No configured strong/moderate/weak criteria were satisfied."],
    }
