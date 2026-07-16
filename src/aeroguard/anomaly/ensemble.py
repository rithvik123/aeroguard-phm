"""Transparent score-fusion and voting ensembles."""

from __future__ import annotations

import numpy as np
import pandas as pd


DETECTOR_ORDER = ["pca_reconstruction", "isolation_forest", "one_class_svm"]


def validate_weights(weights: dict[str, float], detectors: list[str] | None = None) -> dict[str, float]:
    detectors = detectors or DETECTOR_ORDER
    missing = [name for name in detectors if name not in weights]
    if missing:
        raise ValueError(f"Missing fusion weights for detectors: {missing}")
    clean = {name: float(weights[name]) for name in detectors}
    if any(value < 0 for value in clean.values()):
        raise ValueError("Fusion weights must be non-negative.")
    if not np.isclose(sum(clean.values()), 1.0, atol=1e-6):
        raise ValueError("Fusion weights must sum to one.")
    return clean


def fuse_scores(
    score_frame: pd.DataFrame,
    score_columns: dict[str, str],
    method: str,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Fuse calibrated detector scores into one transparent ensemble score."""
    matrix = np.column_stack([score_frame[score_columns[name]].to_numpy(dtype=float) for name in DETECTOR_ORDER])
    if method == "mean":
        return matrix.mean(axis=1)
    if method == "median":
        return np.median(matrix, axis=1)
    if method == "max":
        return matrix.max(axis=1)
    if method == "weighted_mean":
        if weights is None:
            raise ValueError("weighted_mean requires weights.")
        clean = validate_weights(weights)
        vector = np.array([clean[name] for name in DETECTOR_ORDER], dtype=float)
        return matrix @ vector
    if method == "rank_average":
        ranks = np.column_stack(
            [
                pd.Series(matrix[:, idx]).rank(method="average", pct=True).to_numpy(dtype=float)
                for idx in range(matrix.shape[1])
            ]
        )
        return ranks.mean(axis=1)
    raise ValueError(f"Unsupported fusion method: {method}")


def voting_flags(flag_frame: pd.DataFrame, flag_columns: dict[str, str], rule: str) -> np.ndarray:
    """Apply binary voting to detector flags."""
    matrix = np.column_stack([flag_frame[flag_columns[name]].astype(bool).to_numpy() for name in DETECTOR_ORDER])
    counts = matrix.sum(axis=1)
    if rule == "any_one":
        return counts >= 1
    if rule == "at_least_two":
        return counts >= 2
    if rule == "all_three":
        return counts == 3
    raise ValueError("Voting rule must be any_one, at_least_two, or all_three.")


def voting_score(flag_frame: pd.DataFrame, flag_columns: dict[str, str]) -> np.ndarray:
    """Return fraction of detectors voting anomalous."""
    matrix = np.column_stack([flag_frame[flag_columns[name]].astype(bool).to_numpy() for name in DETECTOR_ORDER])
    return matrix.mean(axis=1)
