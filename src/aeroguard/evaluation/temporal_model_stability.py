"""Stability analysis for temporal RUL model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_model_stability(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for model_id, group in metrics.groupby("model_id"):
        rmse = group["validation_rmse"].to_numpy(dtype=float)
        mae = group["validation_mae"].to_numpy(dtype=float)
        nasa = group["validation_nasa_score"].to_numpy(dtype=float)
        optimistic = group["validation_optimistic_rate"].to_numpy(dtype=float)
        best = group.loc[group["validation_rmse"].idxmin()]
        worst = group.loc[group["validation_rmse"].idxmax()]
        q75, q25 = np.quantile(rmse, [0.75, 0.25])
        rows.append(
            {
                "model_id": model_id,
                "run_count": int(len(group)),
                "mean_rmse": float(np.mean(rmse)),
                "median_rmse": float(np.median(rmse)),
                "std_rmse": float(np.std(rmse, ddof=0)),
                "iqr_rmse": float(q75 - q25),
                "mean_mae": float(np.mean(mae)),
                "mean_nasa_score": float(np.mean(nasa)),
                "mean_optimistic_rate": float(np.mean(optimistic)),
                "best_fold": str(best["fold"]),
                "best_seed": int(best["seed"]),
                "best_rmse": float(best["validation_rmse"]),
                "worst_fold": str(worst["fold"]),
                "worst_seed": int(worst["seed"]),
                "worst_rmse": float(worst["validation_rmse"]),
                "robust_selection_score": float(np.mean(rmse) + 0.5 * np.std(rmse, ddof=0) + 0.02 * np.mean(nasa) + 10.0 * np.mean(optimistic)),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_rmse", "std_rmse", "mean_nasa_score", "mean_optimistic_rate", "model_id"]).reset_index(drop=True)


def locked_epoch_from_cv(metrics: pd.DataFrame, model_id: str, maximum_epoch_cap: int | None = None) -> dict[str, int | float]:
    group = metrics[metrics["model_id"] == model_id]
    if group.empty:
        raise ValueError(f"No CV metrics found for model_id={model_id}.")
    epochs = group["best_epoch"].to_numpy(dtype=float)
    q75, q25 = np.quantile(epochs, [0.75, 0.25])
    locked = int(max(1, round(float(np.median(epochs)))))
    if maximum_epoch_cap is not None:
        locked = min(locked, int(maximum_epoch_cap))
    return {
        "median_best_epoch": float(np.median(epochs)),
        "iqr_best_epoch": float(q75 - q25),
        "minimum_best_epoch": int(np.min(epochs)),
        "maximum_best_epoch": int(np.max(epochs)),
        "locked_epoch_count": int(locked),
    }

