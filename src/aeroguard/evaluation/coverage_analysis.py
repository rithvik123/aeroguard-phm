"""Grouped coverage and bootstrap analysis for uncertainty outputs."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def coverage_by_group(frame: pd.DataFrame, group_column: str, levels: list[float]) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(group_column, dropna=False):
        row = {
            group_column: value,
            "engine_count": int(len(group)),
            "mae": float(group["absolute_error"].mean()) if len(group) else None,
            "rmse": float(np.sqrt(np.mean(np.square(group["residual"])))) if len(group) else None,
            "abstention_rate": float(group["abstain_flag"].astype(bool).mean()) if len(group) else None,
        }
        for level in levels:
            pct = int(round(level * 100))
            row[f"coverage_{pct}"] = float(group[f"covered_{pct}"].astype(bool).mean()) if len(group) else None
            row[f"mean_interval_width_{pct}"] = float(group[f"interval_width_{pct}"].mean()) if len(group) else None
            row[f"median_interval_width_{pct}"] = float(group[f"interval_width_{pct}"].median()) if len(group) else None
        rows.append(row)
    return pd.DataFrame(rows)


def assign_numeric_band(values: pd.Series, bands: list[dict[str, float | str]], output_name: str) -> pd.Series:
    labels = []
    for value in values.astype(float):
        label = str(bands[-1]["label"])
        for band in bands:
            upper = band.get("upper")
            upper_value = np.inf if upper is None else float(upper)
            if float(band["lower"]) <= value <= upper_value:
                label = str(band["label"])
                break
        labels.append(label)
    return pd.Series(labels, index=values.index, name=output_name)


def bootstrap_engine_metrics(
    frame: pd.DataFrame,
    metric_functions: dict[str, Callable[[pd.DataFrame], float | None]],
    n_samples: int,
    confidence_level: float,
    seed: int,
    engine_column: str = "global_engine_id",
) -> dict[str, dict[str, float | int | None]]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1).")
    if frame.empty:
        return {}
    rng = np.random.default_rng(seed)
    engines = np.asarray(sorted(frame[engine_column].unique()), dtype=object)
    grouped = {engine: group.copy() for engine, group in frame.groupby(engine_column)}
    estimates = {name: func(frame) for name, func in metric_functions.items()}
    values = {name: [] for name in metric_functions}
    for _ in range(int(n_samples)):
        chosen = rng.choice(engines, size=len(engines), replace=True)
        sample = pd.concat([grouped[engine] for engine in chosen], ignore_index=True)
        for name, func in metric_functions.items():
            value = func(sample)
            if value is not None and np.isfinite(float(value)):
                values[name].append(float(value))
    alpha = 1.0 - confidence_level
    output = {}
    for name, samples in values.items():
        output[name] = {
            "estimate": None if estimates[name] is None else float(estimates[name]),
            "ci_lower": None if not samples else float(np.percentile(samples, 100 * alpha / 2)),
            "ci_upper": None if not samples else float(np.percentile(samples, 100 * (1 - alpha / 2))),
            "valid_replicates": int(len(samples)),
            "bootstrap_samples": int(n_samples),
            "confidence_level": float(confidence_level),
        }
    return output
