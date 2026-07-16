"""Engine-wise onset summaries and test RUL trajectory derivation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aeroguard.data.columns import CYCLE_COLUMN, TEST_TARGET_COLUMN, UNIT_COLUMN
from aeroguard.data.validation import CMapssValidationError, validate_test_rul_alignment
from aeroguard.onset.page_hinkley import PageHinkley


def derive_test_true_rul_trajectory(
    test_frame: pd.DataFrame,
    test_rul: pd.Series,
    output_column: str = "true_rul_uncapped",
) -> pd.DataFrame:
    """Derive per-cycle test RUL for evaluation only.

    For each engine: final RUL from the RUL file plus
    ``max_observed_cycle - current_cycle``.
    """
    validate_test_rul_alignment(test_frame, test_rul)
    result = test_frame.copy()
    ordered_units = sorted(result[UNIT_COLUMN].unique())
    rul_map = {unit_id: float(test_rul.iloc[idx]) for idx, unit_id in enumerate(ordered_units)}
    max_cycle = result.groupby(UNIT_COLUMN)[CYCLE_COLUMN].transform("max")
    result[output_column] = result[UNIT_COLUMN].map(rul_map) + (max_cycle - result[CYCLE_COLUMN])
    final_rows = result.sort_values([UNIT_COLUMN, CYCLE_COLUMN]).groupby(UNIT_COLUMN).tail(1)
    if not np.allclose(final_rows[output_column].to_numpy(), test_rul.to_numpy(dtype=float)):
        raise CMapssValidationError("Derived final test RUL does not match RUL file.")
    return result


def add_proxy_labels(
    frame: pd.DataFrame,
    healthy_rul_threshold: float,
    critical_rul_threshold: float,
    rul_column: str = "true_rul_uncapped",
) -> pd.DataFrame:
    """Add proxy degradation and critical labels from true uncapped RUL."""
    if healthy_rul_threshold <= critical_rul_threshold:
        raise ValueError("healthy_rul_threshold must be greater than critical_rul_threshold.")
    result = frame.copy()
    result["proxy_degradation_label"] = (result[rul_column] <= healthy_rul_threshold).astype(int)
    result["proxy_critical_label"] = (result[rul_column] <= critical_rul_threshold).astype(int)
    result["proxy_health_region"] = np.select(
        [
            result[rul_column] > healthy_rul_threshold,
            result[rul_column] <= critical_rul_threshold,
        ],
        ["healthy_proxy", "critical_proxy"],
        default="degradation_proxy",
    )
    return result


def apply_page_hinkley_by_engine(
    frame: pd.DataFrame,
    signal_column: str,
    output_prefix: str,
    delta: float,
    threshold: float,
    min_observations: int,
    direction: str,
    reset_after_detection: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Page-Hinkley independently to each engine."""
    if signal_column not in frame.columns:
        raise ValueError(f"Missing Page-Hinkley signal column: {signal_column}")
    flag_col = f"{output_prefix}_change_flag"
    first_col = f"{output_prefix}_first_change_cycle"
    result = frame.copy()
    result[flag_col] = False
    result[first_col] = pd.NA
    summary_rows = []
    for unit_id, group in result.sort_values([UNIT_COLUMN, CYCLE_COLUMN]).groupby(UNIT_COLUMN, sort=False):
        detector = PageHinkley(
            delta=delta,
            threshold=threshold,
            min_observations=min_observations,
            direction=direction,
            reset_after_detection=reset_after_detection,
        )
        first_cycle = None
        for index, row in group.iterrows():
            changed = detector.update(float(row[signal_column]))
            result.loc[index, flag_col] = changed
            if changed and first_cycle is None:
                first_cycle = int(row[CYCLE_COLUMN])
        if first_cycle is not None:
            result.loc[result[UNIT_COLUMN] == unit_id, first_col] = first_cycle
        summary_rows.append(
            {
                UNIT_COLUMN: int(unit_id),
                "estimated_onset_cycle": first_cycle,
                "detected": first_cycle is not None,
                "method": output_prefix,
                "signal_column": signal_column,
            }
        )
    return result, pd.DataFrame(summary_rows)
