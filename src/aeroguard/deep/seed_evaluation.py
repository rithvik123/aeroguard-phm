"""Seed-level aggregation helpers for Phase 5B."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_seed_metrics(metrics: pd.DataFrame, group_columns: list[str] | None = None) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    group_columns = group_columns or ["model_id"]
    rows = []
    for keys, group in metrics.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        for metric in ["validation_rmse", "validation_mae", "validation_nasa_score", "validation_optimistic_rate"]:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=0))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        best = group.loc[group["validation_rmse"].idxmin()]
        worst = group.loc[group["validation_rmse"].idxmax()]
        row["best_seed"] = int(best["seed"])
        row["worst_seed"] = int(worst["seed"])
        row["run_count"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_disagreement(predictions: pd.DataFrame, id_columns: list[str] | None = None) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    id_columns = id_columns or ["model_id", "fold", "global_engine_id", "cycle"]
    available = [column for column in id_columns if column in predictions.columns]
    rows = []
    for keys, group in predictions.groupby(available, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["predicted_rul"].to_numpy(dtype=float)
        row = dict(zip(available, keys))
        row["seed_prediction_count"] = int(len(values))
        row["prediction_mean"] = float(np.mean(values))
        row["prediction_std"] = float(np.std(values, ddof=0))
        row["prediction_range"] = float(np.max(values) - np.min(values))
        rows.append(row)
    return pd.DataFrame(rows)

