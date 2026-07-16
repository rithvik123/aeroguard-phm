"""Cross-fold policy summaries and generalization classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SUMMARY_METRICS = [
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "healthy_region_false_positive_rate",
    "critical_region_recall",
    "pr_auc",
    "roc_auc",
    "engine_detection_rate",
    "missed_engine_rate",
    "false_alarm_engine_rate",
    "median_detection_delay",
    "median_lead_time",
    "detected_before_60_fraction",
    "detected_before_30_fraction",
    "detected_before_critical_fraction",
    "median_alert_transitions",
    "mean_alert_transitions",
    "unstable_engine_fraction",
    "no_alert_engine_fraction",
    "multiple_alert_entry_fraction",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def compute_profile_utility(row: pd.Series | dict[str, Any], weights: dict[str, float]) -> float:
    """Compute transparent profile utility for one fold or aggregate row."""
    return float(
        safe_float(weights.get("detection_rate")) * safe_float(row.get("engine_detection_rate"))
        + safe_float(weights.get("critical_region_recall")) * safe_float(row.get("critical_region_recall"))
        + safe_float(weights.get("detected_before_30_fraction")) * safe_float(row.get("detected_before_30_fraction"))
        + safe_float(weights.get("detected_before_60_fraction")) * safe_float(row.get("detected_before_60_fraction"))
        - safe_float(weights.get("missed_engine_rate")) * safe_float(row.get("missed_engine_rate"))
        - safe_float(weights.get("false_alarm_engine_rate")) * safe_float(row.get("false_alarm_engine_rate"))
        - safe_float(weights.get("healthy_region_false_positive_rate")) * safe_float(row.get("healthy_region_false_positive_rate"))
        - safe_float(weights.get("alert_instability")) * safe_float(row.get("unstable_engine_fraction"))
    )


def summarize_policy_folds(
    fold_metrics: pd.DataFrame,
    operational_profiles: dict[str, dict[str, float]],
    variability_penalty: float,
    feasibility_constraints: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate fold metrics and return policy summary and ranking tables."""
    if fold_metrics.empty:
        raise ValueError("fold_metrics must not be empty.")
    rows: list[dict[str, Any]] = []
    utility_columns = []
    enriched = fold_metrics.copy()
    for profile, weights in operational_profiles.items():
        column = f"utility_{profile}"
        utility_columns.append(column)
        enriched[column] = enriched.apply(lambda row: compute_profile_utility(row, weights), axis=1)

    for policy_id, group in enriched.groupby("policy_id"):
        row: dict[str, Any] = {"policy_id": policy_id, "fold_count": int(len(group))}
        for metric in SUMMARY_METRICS:
            if metric not in group.columns:
                values = pd.Series(dtype=float)
            else:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                for suffix in ["mean", "std", "median", "min", "max", "p05", "p95"]:
                    row[f"{metric}_{suffix}"] = None
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
            row[f"{metric}_p05"] = float(np.percentile(values, 5))
            row[f"{metric}_p95"] = float(np.percentile(values, 95))
        row["engine_detection_rate_range"] = (
            None
            if row.get("engine_detection_rate_min") is None
            else float(row["engine_detection_rate_max"] - row["engine_detection_rate_min"])
        )
        row["false_alarm_engine_rate_range"] = (
            None
            if row.get("false_alarm_engine_rate_min") is None
            else float(row["false_alarm_engine_rate_max"] - row["false_alarm_engine_rate_min"])
        )
        failed = failed_constraints(row, feasibility_constraints)
        row["feasible"] = len(failed) == 0
        row["failed_constraints"] = "; ".join(failed)
        for profile in operational_profiles:
            values = pd.to_numeric(group[f"utility_{profile}"], errors="coerce").dropna()
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            row[f"mean_utility_{profile}"] = mean
            row[f"std_utility_{profile}"] = std
            row[f"robust_utility_{profile}"] = mean - float(variability_penalty) * std
        rows.append(row)
    summary = pd.DataFrame(rows)
    ranking = summary.copy()
    for profile in operational_profiles:
        ranking[f"rank_{profile}"] = ranking[f"robust_utility_{profile}"].rank(method="min", ascending=False)
    return summary, ranking


def failed_constraints(summary_row: dict[str, Any], constraints: dict[str, float]) -> list[str]:
    failed = []
    checks = {
        "max_mean_false_alarm_engine_rate": ("false_alarm_engine_rate_mean", lambda value, limit: value <= limit),
        "max_mean_healthy_region_false_positive_rate": (
            "healthy_region_false_positive_rate_mean",
            lambda value, limit: value <= limit,
        ),
        "minimum_mean_detection_rate": ("engine_detection_rate_mean", lambda value, limit: value >= limit),
        "minimum_mean_critical_region_recall": ("critical_region_recall_mean", lambda value, limit: value >= limit),
    }
    for key, limit in constraints.items():
        if key not in checks:
            raise ValueError(f"Unsupported feasibility constraint: {key}")
        metric, predicate = checks[key]
        value = summary_row.get(metric)
        if value is None or not predicate(float(value), float(limit)):
            failed.append(f"{key} failed: {metric}={value}, limit={limit}")
    return failed


def select_locked_policy(summary: pd.DataFrame, primary_profile: str) -> dict[str, Any]:
    """Select one policy using the configured deterministic tie-break order."""
    if summary.empty:
        raise ValueError("Policy summary is empty.")
    robust_col = f"robust_utility_{primary_profile}"
    mean_col = f"mean_utility_{primary_profile}"
    if robust_col not in summary.columns:
        raise ValueError(f"Missing profile in summary: {primary_profile}")
    ranked = summary.copy()
    ranked["_feasible_sort"] = ranked["feasible"].astype(bool).astype(int)
    ranked = ranked.sort_values(
        [
            "_feasible_sort",
            robust_col,
            mean_col,
            "false_alarm_engine_rate_mean",
            "missed_engine_rate_mean",
            "mean_alert_transitions_mean",
            "policy_id",
        ],
        ascending=[False, False, False, True, True, True, True],
    )
    return ranked.iloc[0].drop(labels=["_feasible_sort"]).to_dict()


def classify_generalization(
    criteria: dict[str, dict[str, float]],
    cv_summary: dict[str, Any],
    fd001_metrics: dict[str, Any],
    fd003_metrics: dict[str, Any],
    bootstrap_intervals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify transfer using predeclared threshold criteria."""
    values = {
        "cv_mean_detection_rate": safe_float(cv_summary.get("engine_detection_rate_mean")),
        "cv_detection_rate_std": safe_float(cv_summary.get("engine_detection_rate_std")),
        "fd001_detection_rate": safe_float(fd001_metrics["engine_level"].get("detection_rate")),
        "fd003_detection_rate": safe_float(fd003_metrics["engine_level"].get("detection_rate")),
        "fd003_false_alarm_engine_rate": safe_float(fd003_metrics["engine_level"].get("false_alarm_engine_rate")),
        "fd003_missed_engine_rate": safe_float(fd003_metrics["engine_level"].get("missed_engine_rate")),
        "fd003_critical_region_recall": safe_float(fd003_metrics["row_level"].get("critical_region_recall")),
    }
    for label in ["strong", "moderate", "weak"]:
        thresholds = criteria.get(label, {})
        passed = True
        failed = []
        for key, limit in thresholds.items():
            value = values.get(key)
            if value is None:
                passed = False
                failed.append(f"{key}=missing")
            elif "false_alarm" in key or "missed" in key or key.endswith("_std"):
                if value > float(limit):
                    passed = False
                    failed.append(f"{key}={value} > {limit}")
            elif value < float(limit):
                passed = False
                failed.append(f"{key}={value} < {limit}")
        if passed:
            return {
                "classification": f"{label.title()} generalization",
                "criteria_values": values,
                "failed_criteria": failed,
                "bootstrap_used_for_context": bootstrap_intervals is not None,
            }
    return {
        "classification": "Failed external transfer",
        "criteria_values": values,
        "failed_criteria": ["No configured strong/moderate/weak criteria were satisfied."],
        "bootstrap_used_for_context": bootstrap_intervals is not None,
    }
