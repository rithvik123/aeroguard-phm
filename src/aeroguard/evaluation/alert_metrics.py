"""Alert stability and transition metrics."""

from __future__ import annotations

import pandas as pd

from aeroguard.data.columns import UNIT_COLUMN


def alert_transition_metrics(
    frame: pd.DataFrame,
    state_column: str,
    level_column: str | None = None,
) -> dict[str, int | float]:
    """Summarize alert-state and optional level transitions."""
    transitions = 0
    active_rows = int(frame[state_column].astype(bool).sum())
    for _, group in frame.groupby(UNIT_COLUMN):
        states = group[state_column].astype(bool).tolist()
        transitions += sum(1 for left, right in zip(states, states[1:]) if left != right)
    level_transitions = 0
    if level_column is not None and level_column in frame.columns:
        for _, group in frame.groupby(UNIT_COLUMN):
            levels = group[level_column].tolist()
            level_transitions += sum(1 for left, right in zip(levels, levels[1:]) if left != right)
    engine_count = int(frame[UNIT_COLUMN].nunique())
    return {
        "engine_count": engine_count,
        "active_alert_rows": active_rows,
        "state_transition_count": int(transitions),
        "level_transition_count": int(level_transitions),
        "mean_state_transitions_per_engine": float(transitions / engine_count) if engine_count else 0.0,
        "mean_level_transitions_per_engine": float(level_transitions / engine_count) if engine_count else 0.0,
    }
