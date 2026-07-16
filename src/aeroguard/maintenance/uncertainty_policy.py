"""Demonstration maintenance recommendation policy from uncertainty bounds."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DISCLAIMER = "Demonstration decision-support output; not approved aircraft-maintenance instruction."


def validate_thresholds(thresholds: dict[str, float]) -> None:
    urgent = float(thresholds["urgent_review_max"])
    schedule = float(thresholds["schedule_maintenance_max"])
    plan = float(thresholds["plan_inspection_max"])
    if not (urgent < schedule < plan):
        raise ValueError("Maintenance thresholds must satisfy urgent < schedule < plan.")


def assign_maintenance_recommendations(frame: pd.DataFrame, thresholds: dict[str, float], lower_bound_column: str = "lower_90") -> pd.DataFrame:
    validate_thresholds(thresholds)
    result = frame.copy()
    lower = result[lower_bound_column].astype(float)
    actions = np.full(len(result), "CONTINUE_MONITORING", dtype=object)
    actions[(lower <= float(thresholds["plan_inspection_max"])) & (lower > float(thresholds["schedule_maintenance_max"]))] = "PLAN_INSPECTION"
    actions[(lower <= float(thresholds["schedule_maintenance_max"])) & (lower > float(thresholds["urgent_review_max"]))] = "SCHEDULE_MAINTENANCE"
    actions[lower <= float(thresholds["urgent_review_max"])] = "URGENT_ENGINEERING_REVIEW"
    actions[result["abstain_flag"].astype(bool).to_numpy()] = "ENGINEERING_REVIEW_REQUIRED"
    result["maintenance_action"] = actions
    result["conservative_rul_bound"] = lower
    result["nominal_interval_level"] = 0.90
    result["action_basis"] = np.where(result["abstain_flag"], "abstention override", "lower 90% RUL bound")
    result["maintenance_disclaimer"] = DISCLAIMER
    return result


def maintenance_policy_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    critical = frame["true_rul"].astype(float) <= 15.0
    urgent = frame["maintenance_action"].isin(["URGENT_ENGINEERING_REVIEW", "ENGINEERING_REVIEW_REQUIRED"])
    rows = []
    for action, group in frame.groupby("maintenance_action"):
        rows.append(
            {
                "maintenance_action": action,
                "engine_count": int(len(group)),
                "mean_true_rul": float(group["true_rul"].mean()),
                "median_true_rul": float(group["true_rul"].median()),
                "critical_engine_count": int((group["true_rul"] <= 15.0).sum()),
            }
        )
    return {
        "disclaimer": DISCLAIMER,
        "action_counts": frame["maintenance_action"].value_counts().sort_index().to_dict(),
        "action_distribution_by_subset": frame.groupby(["subset", "maintenance_action"]).size().to_dict(),
        "action_true_rul_summary": rows,
        "critical_engines_not_receiving_urgent_review": int((critical & ~urgent).sum()),
        "engines_true_rul_gt_60_receiving_urgent_review": int(((frame["true_rul"] > 60.0) & urgent).sum()),
        "abstention_count": int(frame["abstain_flag"].sum()),
        "urgent_review_recall_true_rul_le_15": None if int(critical.sum()) == 0 else float((critical & urgent).sum() / critical.sum()),
        "conservative_action_count": int((frame["predicted_rul"] < frame["true_rul"]).sum()),
        "overly_optimistic_action_count": int((frame["predicted_rul"] > frame["true_rul"]).sum()),
    }
