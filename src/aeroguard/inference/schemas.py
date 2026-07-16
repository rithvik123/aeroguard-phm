"""Shared schemas for AeroGuard-PHM inference inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ENGINE_ID_COLUMNS = ["engine_id", "global_engine_id", "unit_id"]
OPERATIONAL_SETTING_COLUMNS = [
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
]
SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]
REQUIRED_INPUT_COLUMNS = ["cycle", *OPERATIONAL_SETTING_COLUMNS, *SENSOR_COLUMNS]
OPTIONAL_INPUT_COLUMNS = ["engine_id", "global_engine_id", "unit_id", "subset"]
EXPECTED_INPUT_COLUMNS = [*OPTIONAL_INPUT_COLUMNS, *REQUIRED_INPUT_COLUMNS]

OUTPUT_FIELDS = [
    "engine_id",
    "model_version",
    "base_rul",
    "safety_adjusted_rul",
    "correction_cycles",
    "lower_80",
    "upper_80",
    "lower_90",
    "upper_90",
    "lower_95",
    "upper_95",
    "interval_width_90",
    "operating_regime",
    "support_status",
    "support_score",
    "safety_guard_activated",
    "review_required",
    "maintenance_action",
    "warnings",
    "explanation",
]


@dataclass(frozen=True)
class ValidationIssue:
    """Structured input validation issue."""

    code: str
    message: str
    severity: str = "error"
    column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "column": self.column,
        }


def output_schema() -> dict[str, str]:
    """Return a concise public output schema."""

    return {
        "engine_id": "Input engine identifier when provided, otherwise a deterministic fallback.",
        "model_version": "Frozen model version string.",
        "base_rul": "Backbone RUL estimate before the critical-boundary guard.",
        "safety_adjusted_rul": "RUL after the frozen safety guard.",
        "correction_cycles": "Downward correction applied by the safety guard.",
        "lower_80": "Lower 80% conformal interval bound.",
        "upper_80": "Upper 80% conformal interval bound.",
        "lower_90": "Lower 90% conformal interval bound.",
        "upper_90": "Upper 90% conformal interval bound.",
        "lower_95": "Lower 95% conformal interval bound.",
        "upper_95": "Upper 95% conformal interval bound.",
        "interval_width_90": "Width of the 90% conformal interval.",
        "operating_regime": "Operating-regime assignment when available.",
        "support_status": "supported, limited_support, or unsupported.",
        "support_score": "Heuristic support score in [0, 1] from validation and range checks.",
        "safety_guard_activated": "Whether the critical-boundary guard fired.",
        "review_required": "Whether the output requires engineering review under the frozen policy.",
        "maintenance_action": "Frozen maintenance-policy recommendation.",
        "warnings": "Structured validation/runtime warnings.",
        "explanation": "Short explanation strings for the prediction and policy decision.",
    }
