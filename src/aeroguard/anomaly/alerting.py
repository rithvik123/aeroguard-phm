"""Persistence, hysteresis, and operational alert levels."""

from __future__ import annotations

import pandas as pd

from aeroguard.data.columns import CYCLE_COLUMN, UNIT_COLUMN


def _is_gap(previous_cycle: int | None, cycle: int) -> bool:
    return previous_cycle is not None and cycle != previous_cycle + 1


def apply_persistence_rule(
    frame: pd.DataFrame,
    flag_column: str,
    score_column: str,
    output_prefix: str,
    rule: dict,
    unit_column: str = UNIT_COLUMN,
    cycle_column: str = CYCLE_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply consecutive, K-of-N, or score-duration persistence by engine."""
    kind = rule.get("type")
    if kind not in {"consecutive", "k_of_n", "score_duration"}:
        raise ValueError("Persistence rule type must be consecutive, k_of_n, or score_duration.")
    result = frame.copy()
    state_col = f"{output_prefix}_persistent_alarm_state"
    started_col = f"{output_prefix}_persistent_alarm_started"
    duration_col = f"{output_prefix}_alarm_duration"
    first_col = f"{output_prefix}_first_alarm_cycle"
    cleared_col = f"{output_prefix}_persistent_alarm_cleared"
    transition_col = f"{output_prefix}_alarm_transition"
    result[state_col] = False
    result[started_col] = False
    result[cleared_col] = False
    result[duration_col] = 0
    result[first_col] = pd.NA
    result[transition_col] = ""
    summary_rows: list[dict[str, object]] = []

    for unit_id, group in result.sort_values([unit_column, cycle_column]).groupby(unit_column, sort=False):
        state = False
        first_cycle = None
        transitions = 0
        cleared = False
        reappeared = False
        duration = 0
        recent: list[tuple[int, bool]] = []
        consecutive = 0
        previous_cycle = None
        first_rul = None

        for index, row in group.iterrows():
            cycle = int(row[cycle_column])
            gap = _is_gap(previous_cycle, cycle)
            flag = bool(row[flag_column])
            score = float(row[score_column])
            qualifies = False
            if kind == "consecutive":
                required = int(rule["k"])
                consecutive = 1 if flag and gap else (consecutive + 1 if flag else 0)
                qualifies = consecutive >= required
            elif kind == "k_of_n":
                k = int(rule["k"])
                n = int(rule["n"])
                recent = [] if gap else recent
                recent.append((cycle, flag))
                recent = recent[-n:]
                span_ok = len(recent) == n and recent[-1][0] - recent[0][0] == n - 1
                qualifies = span_ok and sum(item[1] for item in recent) >= k
            else:
                required = int(rule["duration"])
                threshold = float(rule["threshold"])
                high = score >= threshold
                consecutive = 1 if high and gap else (consecutive + 1 if high else 0)
                qualifies = consecutive >= required

            if qualifies and not state:
                state = True
                transitions += 1
                result.loc[index, started_col] = True
                result.loc[index, transition_col] = "start"
                if first_cycle is None:
                    if kind == "k_of_n":
                        first_cycle = int(recent[0][0])
                    else:
                        first_cycle = cycle - int(rule.get("k", rule.get("duration", 1))) + 1
                    if "true_rul_uncapped" in group.columns:
                        match = group[group[cycle_column] == first_cycle]
                        if not match.empty:
                            first_rul = float(match["true_rul_uncapped"].iloc[0])
                elif cleared:
                    reappeared = True
            elif not qualifies and state:
                state = False
                transitions += 1
                cleared = True
                result.loc[index, cleared_col] = True
                result.loc[index, transition_col] = "clear"
            result.loc[index, state_col] = state
            if state:
                duration += 1
                result.loc[index, duration_col] = duration
            previous_cycle = cycle

        if first_cycle is not None:
            result.loc[result[unit_column] == unit_id, first_col] = first_cycle
        summary_rows.append(
            {
                unit_column: int(unit_id),
                "rule_name": rule.get("name", output_prefix),
                "first_persistent_alarm_cycle": first_cycle,
                "first_persistent_alarm_rul": first_rul,
                "alarm_duration": int(duration),
                "number_of_alarm_transitions": int(transitions),
                "alarm_cleared": bool(cleared),
                "alarm_reappeared": bool(reappeared),
            }
        )
    return result, pd.DataFrame(summary_rows)


def apply_hysteresis_alert(
    frame: pd.DataFrame,
    score_column: str,
    output_prefix: str,
    enter_threshold: float,
    exit_threshold: float,
    min_enter_duration: int,
    min_clear_duration: int,
    unit_column: str = UNIT_COLUMN,
    cycle_column: str = CYCLE_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply score-based alert hysteresis independently per engine."""
    if exit_threshold >= enter_threshold:
        raise ValueError("exit_threshold must be less than enter_threshold.")
    if min_enter_duration <= 0 or min_clear_duration <= 0:
        raise ValueError("Minimum enter/clear durations must be positive.")
    result = frame.copy()
    state_col = f"{output_prefix}_alert_state"
    started_col = f"{output_prefix}_alert_started"
    cleared_col = f"{output_prefix}_alert_cleared"
    transition_col = f"{output_prefix}_alert_transition"
    first_col = f"{output_prefix}_first_alert_cycle"
    transitions_col = f"{output_prefix}_number_of_transitions"
    result[state_col] = False
    result[started_col] = False
    result[cleared_col] = False
    result[transition_col] = ""
    result[first_col] = pd.NA
    result[transitions_col] = 0
    summary_rows = []

    for unit_id, group in result.sort_values([unit_column, cycle_column]).groupby(unit_column, sort=False):
        state = False
        enter_run = 0
        clear_run = 0
        transitions = 0
        first_alert_cycle = None
        previous_cycle = None
        for index, row in group.iterrows():
            cycle = int(row[cycle_column])
            gap = _is_gap(previous_cycle, cycle)
            score = float(row[score_column])
            if gap:
                enter_run = 0
                clear_run = 0
            if not state:
                enter_run = enter_run + 1 if score >= enter_threshold else 0
                if enter_run >= min_enter_duration:
                    state = True
                    transitions += 1
                    start_cycle = cycle - min_enter_duration + 1
                    if first_alert_cycle is None:
                        first_alert_cycle = start_cycle
                    result.loc[index, started_col] = True
                    result.loc[index, transition_col] = "start"
                    clear_run = 0
            else:
                clear_run = clear_run + 1 if score <= exit_threshold else 0
                if clear_run >= min_clear_duration:
                    state = False
                    transitions += 1
                    result.loc[index, cleared_col] = True
                    result.loc[index, transition_col] = "clear"
                    enter_run = 0
            result.loc[index, state_col] = state
            previous_cycle = cycle
        if first_alert_cycle is not None:
            result.loc[result[unit_column] == unit_id, first_col] = first_alert_cycle
        result.loc[result[unit_column] == unit_id, transitions_col] = transitions
        summary_rows.append(
            {
                unit_column: int(unit_id),
                "first_alert_cycle": first_alert_cycle,
                "number_of_transitions": int(transitions),
                "alert_ever_active": first_alert_cycle is not None,
            }
        )
    return result, pd.DataFrame(summary_rows)


def assign_operational_alert_levels(
    frame: pd.DataFrame,
    score_column: str,
    persistent_column: str,
    health_column: str,
    output_column: str,
    thresholds: dict,
) -> pd.DataFrame:
    """Assign NORMAL/MONITOR/WARNING/CRITICAL without using proxy labels."""
    result = frame.copy()
    monitor = float(thresholds["monitor_score"])
    warning = float(thresholds["warning_score"])
    critical = float(thresholds["critical_score"])
    warning_hi = float(thresholds["warning_health_index_max"])
    critical_hi = float(thresholds["critical_health_index_max"])

    levels = []
    for _, row in result.iterrows():
        score = float(row[score_column])
        persistent = bool(row[persistent_column])
        health = float(row[health_column])
        if persistent and (score >= critical or health <= critical_hi):
            levels.append("CRITICAL")
        elif persistent or (score >= warning and health <= warning_hi):
            levels.append("WARNING")
        elif score >= monitor:
            levels.append("MONITOR")
        else:
            levels.append("NORMAL")
    result[output_column] = levels
    return result
