"""One-Class SVM anomaly detector wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.svm import OneClassSVM


@dataclass
class OneClassSVMAnomalyDetector:
    """One-Class SVM with deterministic healthy-row subsampling."""

    kernel: str = "rbf"
    nu: float = 0.05
    gamma: str | float = "scale"
    max_training_rows: int | None = 3000
    random_state: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < float(self.nu) < 1.0:
            raise ValueError("nu must be in (0, 1).")
        if self.max_training_rows is not None and self.max_training_rows <= 0:
            raise ValueError("max_training_rows must be positive or null.")
        self.model_: OneClassSVM | None = None
        self.subsampling_applied_: bool = False
        self.fit_row_count_: int = 0

    def _fit_subset(self, healthy_x: np.ndarray) -> np.ndarray:
        if self.max_training_rows is None or len(healthy_x) <= self.max_training_rows:
            self.subsampling_applied_ = False
            return healthy_x
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(healthy_x), size=self.max_training_rows, replace=False)
        self.subsampling_applied_ = True
        return healthy_x[np.sort(indices)]

    def fit(self, healthy_x: np.ndarray) -> "OneClassSVMAnomalyDetector":
        x = np.asarray(healthy_x, dtype=float)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("healthy_x must be a non-empty 2D array.")
        fit_x = self._fit_subset(x)
        self.fit_row_count_ = int(len(fit_x))
        self.model_ = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self.model_.fit(fit_x)
        return self

    def score(self, x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model_ is None:
            raise RuntimeError("Detector must be fitted before scoring.")
        x = np.asarray(x_values, dtype=float)
        scores = -self.model_.decision_function(x)
        flags = self.model_.predict(x) == -1
        return scores, flags
