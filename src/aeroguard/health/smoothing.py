"""Engine-wise causal smoothing for health signals."""

from __future__ import annotations

import pandas as pd

from aeroguard.data.columns import CYCLE_COLUMN, UNIT_COLUMN


def smooth_by_engine(
    frame: pd.DataFrame,
    value_column: str,
    output_column: str,
    method: str = "median",
    window: int = 5,
    causal: bool = True,
    unit_column: str = UNIT_COLUMN,
    cycle_column: str = CYCLE_COLUMN,
) -> pd.DataFrame:
    """Smooth a signal independently within each engine.

    Missing cycle numbers are not filled. With causal smoothing, each row uses
    only the current and previous observed rows for the same engine.
    """
    if window <= 0:
        raise ValueError("Smoothing window must be positive.")
    if method not in {"median", "mean", "none"}:
        raise ValueError("Smoothing method must be 'median', 'mean', or 'none'.")
    if value_column not in frame.columns:
        raise ValueError(f"Missing value column: {value_column}")

    result = frame.copy()
    if method == "none" or window == 1:
        result[output_column] = result[value_column]
        return result

    sorted_frame = result.sort_values([unit_column, cycle_column])
    center = not causal
    smoothed_parts = []
    for _, group in sorted_frame.groupby(unit_column, sort=False):
        rolling = group[value_column].rolling(window=window, min_periods=1, center=center)
        if method == "median":
            values = rolling.median()
        else:
            values = rolling.mean()
        smoothed_parts.append(values)
    smoothed = pd.concat(smoothed_parts).sort_index()
    result[output_column] = smoothed.reindex(result.index)
    return result
