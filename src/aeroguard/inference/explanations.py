"""Short explanation strings for final-release predictions."""

from __future__ import annotations


def build_explanation(
    *,
    guard_active: bool,
    maintenance_action: str,
    interval_width_90: float | None,
    support_status: str,
) -> list[str]:
    explanation: list[str] = []
    if guard_active:
        explanation.append("Critical-boundary safety guard activated")
    if maintenance_action == "URGENT_ENGINEERING_REVIEW":
        explanation.append("Safety-adjusted RUL crossed the urgent review threshold")
    elif maintenance_action == "SCHEDULE_MAINTENANCE":
        explanation.append("Safety-adjusted RUL crossed the scheduled maintenance threshold")
    elif maintenance_action == "PLAN_INSPECTION":
        explanation.append("Safety-adjusted RUL crossed the inspection planning threshold")
    if interval_width_90 is not None and interval_width_90 > 80:
        explanation.append("Prediction interval is wide; interpret the estimate conservatively")
    if support_status != "supported":
        explanation.append(f"Input support status is {support_status}")
    if not explanation:
        explanation.append("No critical-boundary or maintenance-review trigger fired")
    return explanation
