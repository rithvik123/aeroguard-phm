"""Isolation Forest anomaly detector wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestAnomalyDetector:
    """CPU-safe Isolation Forest wrapper with consistent score direction."""

    n_estimators: int = 100
    max_samples: int | float | str = "auto"
    contamination: float | str = "auto"
    random_state: int = 42
    n_jobs: int | None = -1

    def __post_init__(self) -> None:
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        self.model_: IsolationForest | None = None

    def fit(self, healthy_x: np.ndarray) -> "IsolationForestAnomalyDetector":
        x = np.asarray(healthy_x, dtype=float)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("healthy_x must be a non-empty 2D array.")
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(x)
        return self

    def score(self, x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model_ is None:
            raise RuntimeError("Detector must be fitted before scoring.")
        x = np.asarray(x_values, dtype=float)
        scores = -self.model_.decision_function(x)
        flags = self.model_.predict(x) == -1
        return scores, flags
