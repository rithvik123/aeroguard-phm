"""Input validation and lightweight preprocessing for production inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from aeroguard.inference.schemas import (
    ENGINE_ID_COLUMNS,
    EXPECTED_INPUT_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    SENSOR_COLUMNS,
    ValidationIssue,
)


def infer_engine_id(frame: pd.DataFrame) -> str:
    for column in ENGINE_ID_COLUMNS:
        if column in frame.columns and len(frame[column]):
            value = frame[column].iloc[0]
            if pd.notna(value):
                return str(value)
    return "engine_0"


def validate_engine_history(
    frame: pd.DataFrame,
    *,
    min_history: int = 10,
    max_history: int = 500,
    expected_columns: list[str] | None = None,
) -> list[ValidationIssue]:
    """Validate a single engine history without mutating it."""

    issues: list[ValidationIssue] = []
    expected = expected_columns or EXPECTED_INPUT_COLUMNS
    if frame.empty:
        return [ValidationIssue("empty_input", "Engine history is empty.")]

    for column in REQUIRED_INPUT_COLUMNS:
        if column not in frame.columns:
            issues.append(ValidationIssue("missing_column", f"Missing required column: {column}", column=column))

    unknown = [column for column in frame.columns if column not in expected]
    for column in unknown:
        issues.append(ValidationIssue("unknown_column", f"Unknown input column: {column}", severity="warning", column=column))

    present_required = [column for column in REQUIRED_INPUT_COLUMNS if column in frame.columns]
    if present_required:
        actual_order = [column for column in frame.columns if column in REQUIRED_INPUT_COLUMNS]
        if actual_order != present_required:
            issues.append(
                ValidationIssue(
                    "column_order_changed",
                    "Required columns are present but not in canonical order.",
                    severity="warning",
                )
            )

    if len(frame) < int(min_history):
        issues.append(
            ValidationIssue(
                "short_history",
                f"Engine history has {len(frame)} rows; minimum expected history is {min_history}.",
                severity="warning",
            )
        )
    if len(frame) > int(max_history):
        issues.append(
            ValidationIssue(
                "long_history",
                f"Engine history has {len(frame)} rows; maximum expected history is {max_history}.",
                severity="warning",
            )
        )

    if "cycle" in frame.columns:
        cycle = pd.to_numeric(frame["cycle"], errors="coerce")
        if cycle.isna().any():
            issues.append(ValidationIssue("invalid_cycle", "Cycle column contains non-numeric values.", column="cycle"))
        if cycle.duplicated().any():
            issues.append(ValidationIssue("duplicate_cycles", "Cycle values must be unique for one engine.", column="cycle"))
        if len(cycle.dropna()) > 1 and not cycle.dropna().is_monotonic_increasing:
            issues.append(ValidationIssue("non_monotonic_cycles", "Cycle values must be monotonically increasing.", column="cycle"))

    for column in present_required:
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.isna().any():
            issues.append(ValidationIssue("missing_or_non_numeric", f"{column} contains missing or non-numeric values.", column=column))
        if not np.isfinite(series.dropna().to_numpy(dtype=float)).all():
            issues.append(ValidationIssue("non_finite_value", f"{column} contains infinite values.", column=column))

    excessive_padding = "cycle" in frame.columns and pd.to_numeric(frame["cycle"], errors="coerce").notna().sum() < max(1, len(frame) // 2)
    if excessive_padding:
        issues.append(ValidationIssue("excessive_padding", "More than half of cycle rows appear invalid.", severity="warning"))

    for column in SENSOR_COLUMNS:
        if column in frame.columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().any():
                low, high = float(numeric.min()), float(numeric.max())
                if high - low > 10000 or low < -10000 or high > 100000:
                    issues.append(
                        ValidationIssue(
                            "sensor_range_warning",
                            f"{column} has values outside conservative expected engineering ranges.",
                            severity="warning",
                            column=column,
                        )
                    )
    return issues


def support_from_issues(issues: list[ValidationIssue]) -> tuple[str, float]:
    error_count = sum(issue.severity == "error" for issue in issues)
    warning_count = sum(issue.severity != "error" for issue in issues)
    if error_count:
        return "unsupported", 0.0
    score = max(0.0, 1.0 - 0.08 * warning_count)
    if score < 0.75:
        return "limited_support", float(score)
    return "supported", float(score)


def validation_payload(issues: list[ValidationIssue]) -> dict[str, Any]:
    return {
        "valid": not any(issue.severity == "error" for issue in issues),
        "errors": [issue.to_dict() for issue in issues if issue.severity == "error"],
        "warnings": [issue.to_dict() for issue in issues if issue.severity != "error"],
    }
