"""Interval calibration helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from aeroguard.uncertainty.conformal import conformalize_interval


def fit_interval_expansion_by_level(frame: pd.DataFrame, nominal_levels: Iterable[float], prefix: str = "") -> dict[float, float]:
    corrections: dict[float, float] = {}
    for level in nominal_levels:
        pct = int(round(float(level) * 100))
        corrections[float(level)] = conformalize_interval(
            frame["true_rul"],
            frame[f"{prefix}lower_{pct}"],
            frame[f"{prefix}upper_{pct}"],
            [float(level)],
        )[float(level)]
    return corrections


def apply_interval_expansion(frame: pd.DataFrame, corrections: dict[float, float], prefix: str = "") -> pd.DataFrame:
    result = frame.copy()
    for level, correction in corrections.items():
        pct = int(round(float(level) * 100))
        result[f"{prefix}lower_{pct}"] = np.maximum(0.0, result[f"{prefix}lower_{pct}"].astype(float) - float(correction))
        result[f"{prefix}upper_{pct}"] = result[f"{prefix}upper_{pct}"].astype(float) + float(correction)
        result[f"{prefix}interval_expansion_{pct}"] = float(correction)
    return result
