"""Frozen maintenance recommendation policy."""

from __future__ import annotations


def recommend_maintenance(adjusted_rul: float, policy: dict[str, object], *, review_required: bool | None = None) -> dict[str, object]:
    urgent = float(policy.get("urgent_threshold", 15.0))
    schedule = float(policy.get("schedule_threshold", 30.0))
    inspect = float(policy.get("inspection_threshold", 60.0))
    value = float(adjusted_rul)
    if value <= urgent:
        action = "URGENT_ENGINEERING_REVIEW"
        review = True
    elif value <= schedule:
        action = "SCHEDULE_MAINTENANCE"
        review = bool(review_required)
    elif value <= inspect:
        action = "PLAN_INSPECTION"
        review = bool(review_required)
    else:
        action = "CONTINUE_MONITORING"
        review = bool(review_required)
    return {
        "maintenance_action": action,
        "review_required": bool(review),
        "policy_id": policy.get("policy_id", "point_u15_s30_i60"),
    }
