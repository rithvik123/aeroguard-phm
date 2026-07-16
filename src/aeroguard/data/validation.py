"""Strict structural validation for NASA C-MAPSS tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aeroguard.data.columns import (
    CMAPSS_COLUMNS,
    CYCLE_COLUMN,
    EXPECTED_CMAPSS_COLUMN_COUNT,
    UNIT_COLUMN,
)


class CMapssValidationError(ValueError):
    """Raised when C-MAPSS input data has a serious structural problem."""


def validate_raw_column_count(column_count: int, source: str) -> None:
    """Validate the meaningful C-MAPSS column count before assigning names."""
    if column_count != EXPECTED_CMAPSS_COLUMN_COUNT:
        raise CMapssValidationError(
            f"{source} has {column_count} meaningful columns; expected "
            f"{EXPECTED_CMAPSS_COLUMN_COUNT} columns."
        )


def _require_numeric(frame: pd.DataFrame, source: str) -> None:
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid_mask = converted.isna() & frame[column].notna()
        if invalid_mask.any():
            first_bad = frame.loc[invalid_mask, column].iloc[0]
            raise CMapssValidationError(
                f"{source} column '{column}' contains a non-numeric value: {first_bad!r}."
            )
        frame[column] = converted


def _require_finite(frame: pd.DataFrame, source: str) -> None:
    if frame.isna().any().any():
        columns = frame.columns[frame.isna().any()].tolist()
        raise CMapssValidationError(
            f"{source} contains missing values in columns: {columns}."
        )
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise CMapssValidationError(f"{source} contains infinite values.")


def _require_positive_integer_series(series: pd.Series, name: str, source: str) -> None:
    values = series.to_numpy(dtype=float)
    if (values <= 0).any():
        raise CMapssValidationError(f"{source} column '{name}' must be positive.")
    if not np.equal(values, np.floor(values)).all():
        raise CMapssValidationError(
            f"{source} column '{name}' must contain integer values."
        )


def validate_cmapss_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate and return a numeric C-MAPSS frame.

    The function refuses structural repairs: invalid column counts, missing
    values, duplicate unit-cycle pairs, and non-monotonic engine trajectories
    raise clear exceptions.
    """
    if list(frame.columns) != CMAPSS_COLUMNS:
        raise CMapssValidationError(
            f"{source} columns do not match the standard C-MAPSS schema."
        )
    if frame.empty:
        raise CMapssValidationError(f"{source} is empty.")

    validated = frame.copy()
    _require_numeric(validated, source)
    _require_finite(validated, source)
    _require_positive_integer_series(validated[UNIT_COLUMN], UNIT_COLUMN, source)
    _require_positive_integer_series(validated[CYCLE_COLUMN], CYCLE_COLUMN, source)

    validated[UNIT_COLUMN] = validated[UNIT_COLUMN].astype(int)
    validated[CYCLE_COLUMN] = validated[CYCLE_COLUMN].astype(int)

    duplicate_mask = validated.duplicated([UNIT_COLUMN, CYCLE_COLUMN])
    if duplicate_mask.any():
        duplicate = validated.loc[duplicate_mask, [UNIT_COLUMN, CYCLE_COLUMN]].iloc[0]
        raise CMapssValidationError(
            f"{source} has a duplicate unit-cycle pair: "
            f"unit_id={duplicate[UNIT_COLUMN]}, cycle={duplicate[CYCLE_COLUMN]}."
        )

    group_sizes = validated.groupby(UNIT_COLUMN, sort=True).size()
    if group_sizes.empty or (group_sizes <= 0).any():
        raise CMapssValidationError(f"{source} contains an empty engine group.")

    for unit_id, group in validated.groupby(UNIT_COLUMN, sort=True):
        cycles = group[CYCLE_COLUMN].to_numpy()
        if not np.all(np.diff(cycles) > 0):
            raise CMapssValidationError(
                f"{source} cycles must be strictly increasing within engine "
                f"{unit_id}."
            )

    return validated


def validate_rul_values(values: pd.Series, source: str) -> pd.Series:
    """Validate one non-negative RUL value per test engine."""
    if values.empty:
        raise CMapssValidationError(f"{source} is empty.")

    converted = pd.to_numeric(values, errors="coerce")
    if converted.isna().any():
        raise CMapssValidationError(f"{source} contains missing or non-numeric RUL.")
    if not np.isfinite(converted.to_numpy(dtype=float)).all():
        raise CMapssValidationError(f"{source} contains infinite RUL values.")
    if (converted < 0).any():
        first_bad = converted[converted < 0].iloc[0]
        raise CMapssValidationError(
            f"{source} contains a negative RUL value: {first_bad}."
        )
    converted.name = "test_final_rul"
    return converted.reset_index(drop=True)


def validate_test_rul_alignment(test_frame: pd.DataFrame, rul_values: pd.Series) -> None:
    """Ensure the RUL file has exactly one row per test engine."""
    engine_count = test_frame[UNIT_COLUMN].nunique()
    if engine_count != len(rul_values):
        raise CMapssValidationError(
            f"Test data has {engine_count} engines but RUL file has "
            f"{len(rul_values)} rows."
        )
