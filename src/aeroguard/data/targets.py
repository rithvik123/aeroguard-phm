"""RUL target generation for AeroGuard C-MAPSS baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aeroguard.data.columns import CYCLE_COLUMN, TEST_TARGET_COLUMN, UNIT_COLUMN
from aeroguard.data.validation import CMapssValidationError, validate_test_rul_alignment


def add_training_rul_targets(
    frame: pd.DataFrame,
    rul_cap: int | float | None = 125,
    unit_column: str = UNIT_COLUMN,
    cycle_column: str = CYCLE_COLUMN,
) -> pd.DataFrame:
    """Add uncapped and capped per-cycle RUL targets to training trajectories."""
    if rul_cap is not None and rul_cap <= 0:
        raise CMapssValidationError("rul_cap must be positive when clipping is enabled.")
    if unit_column not in frame or cycle_column not in frame:
        raise CMapssValidationError(
            f"Training frame must contain '{unit_column}' and '{cycle_column}'."
        )

    result = frame.copy()
    max_cycle = result.groupby(unit_column)[cycle_column].transform("max")
    result["rul_uncapped"] = max_cycle - result[cycle_column]
    if (result["rul_uncapped"] < 0).any():
        raise CMapssValidationError("Computed negative training RUL values.")
    if rul_cap is None:
        result["rul_capped"] = result["rul_uncapped"]
    else:
        result["rul_capped"] = np.minimum(result["rul_uncapped"], rul_cap)

    final_rows = result[cycle_column] == max_cycle
    if not (result.loc[final_rows, "rul_uncapped"] == 0).all():
        raise CMapssValidationError("Final training cycle RUL must be zero.")
    return result


def final_observed_test_rows(
    test_frame: pd.DataFrame,
    test_rul: pd.Series,
    unit_column: str = UNIT_COLUMN,
    cycle_column: str = CYCLE_COLUMN,
    target_column: str = TEST_TARGET_COLUMN,
) -> pd.DataFrame:
    """Return one final observed row per test engine with its true final-cycle RUL."""
    if unit_column not in test_frame or cycle_column not in test_frame:
        raise CMapssValidationError(
            f"Test frame must contain '{unit_column}' and '{cycle_column}'."
        )
    validate_test_rul_alignment(test_frame, test_rul)
    if (pd.to_numeric(test_rul, errors="coerce") < 0).any():
        raise CMapssValidationError("Test RUL values must be non-negative.")

    ordered_units = sorted(test_frame[unit_column].unique())
    final_rows = (
        test_frame.sort_values([unit_column, cycle_column])
        .groupby(unit_column, as_index=False)
        .tail(1)
        .sort_values(unit_column)
        .reset_index(drop=True)
    )
    if final_rows[unit_column].tolist() != ordered_units:
        raise CMapssValidationError("Could not map final test rows to ordered RUL rows.")

    result = final_rows.copy()
    result[target_column] = pd.to_numeric(test_rul, errors="raise").to_numpy(dtype=float)
    return result
