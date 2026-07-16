"""Persistent anomaly-alarm logic."""

from __future__ import annotations

import pandas as pd

from aeroguard.data.columns import CYCLE_COLUMN, UNIT_COLUMN


def apply_persistent_alarms(
    frame: pd.DataFrame,
    flag_column: str,
    output_prefix: str,
    persistence_window: int = 5,
    alarm_state_from_onset: bool = True,
    require_consecutive_cycles: bool = True,
    unit_column: str = UNIT_COLUMN,
    cycle_column: str = CYCLE_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply engine-wise consecutive-anomaly persistence.

    Cycle gaps break a run by default. For example, anomaly flags at cycles
    5 and 7 do not form a two-cycle persistent run unless
    ``require_consecutive_cycles`` is false.
    """
    if persistence_window <= 0:
        raise ValueError("persistence_window must be positive.")
    if flag_column not in frame.columns:
        raise ValueError(f"Missing flag column: {flag_column}")

    flag_out = f"{output_prefix}_persistent_alarm_flag"
    started_out = f"{output_prefix}_persistent_alarm_started"
    first_cycle_out = f"{output_prefix}_first_persistent_alarm_cycle"
    result = frame.copy()
    result[flag_out] = False
    result[started_out] = False
    result[first_cycle_out] = pd.NA
    summary_rows: list[dict[str, object]] = []

    for unit_id, group in result.sort_values([unit_column, cycle_column]).groupby(unit_column, sort=False):
        run_start_index = None
        run_length = 0
        previous_cycle = None
        first_alarm_cycle = None
        first_alarm_index = None

        for index, row in group.iterrows():
            cycle = int(row[cycle_column])
            is_consecutive = (
                previous_cycle is None
                or not require_consecutive_cycles
                or cycle == previous_cycle + 1
            )
            if bool(row[flag_column]) and is_consecutive:
                if run_length == 0:
                    run_start_index = index
                run_length += 1
            elif bool(row[flag_column]):
                run_start_index = index
                run_length = 1
            else:
                run_start_index = None
                run_length = 0

            if first_alarm_cycle is None and run_length >= persistence_window:
                first_alarm_index = run_start_index
                first_alarm_cycle = int(result.loc[first_alarm_index, cycle_column])
                result.loc[first_alarm_index, started_out] = True
                if alarm_state_from_onset:
                    engine_mask = (result[unit_column] == unit_id) & (
                        result[cycle_column] >= first_alarm_cycle
                    )
                    result.loc[engine_mask, flag_out] = True
                else:
                    result.loc[index, flag_out] = True

            previous_cycle = cycle

        if first_alarm_cycle is not None:
            result.loc[result[unit_column] == unit_id, first_cycle_out] = first_alarm_cycle
        summary_rows.append(
            {
                unit_column: int(unit_id),
                "first_persistent_alarm_cycle": first_alarm_cycle,
                "persistent_alarm_detected": first_alarm_cycle is not None,
                "source_flag_column": flag_column,
                "output_prefix": output_prefix,
                "persistence_window": persistence_window,
            }
        )
    return result, pd.DataFrame(summary_rows)
