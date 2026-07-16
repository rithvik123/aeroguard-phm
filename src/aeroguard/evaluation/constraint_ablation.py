"""Constraint-ablation utilities for later physics-guided model selection."""

from __future__ import annotations

from typing import Any

import pandas as pd

from aeroguard.evaluation.physics_guided_rul_metrics import normalized_robust_score


def constraint_ablation_frame(candidate_metrics: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a tabular view of candidate metrics without using benchmark test data."""

    if not candidate_metrics:
        return pd.DataFrame()
    frame = pd.DataFrame(candidate_metrics)
    if "candidate_id" not in frame.columns:
        raise ValueError("candidate_id is required for constraint ablation.")
    return frame.sort_values("candidate_id").reset_index(drop=True)


def rank_candidates_by_training_score(candidate_metrics: list[dict[str, Any]], weights: dict[str, float]) -> pd.DataFrame:
    """Rank candidates using configured training-validation robust-score weights."""

    frame = constraint_ablation_frame(candidate_metrics)
    if frame.empty:
        return frame
    rows = []
    for _, row in frame.iterrows():
        metric_dict = {key: float(row[key]) for key in frame.columns if key != "candidate_id" and pd.notna(row[key])}
        rows.append({"candidate_id": row["candidate_id"], "robust_score": normalized_robust_score(metric_dict, weights)})
    return pd.DataFrame(rows).sort_values(["robust_score", "candidate_id"]).reset_index(drop=True)
