"""PCA reconstruction-error anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class PCAReconstructionAnomalyDetector:
    """Detect anomalies using PCA reconstruction error."""

    n_components: int | float = 0.95
    threshold_percentile: float = 99.0

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold_percentile < 100.0:
            raise ValueError("threshold_percentile must be in (0, 100).")
        self.pca_: PCA | None = None
        self.threshold_: float = 0.0
        self.explained_variance_ratio_: list[float] = []

    def _resolved_components(self, x_values: np.ndarray) -> int | float:
        if isinstance(self.n_components, float):
            if not 0.0 < self.n_components <= 1.0:
                raise ValueError("Float n_components must be in (0, 1].")
            return self.n_components
        requested = int(self.n_components)
        if requested <= 0:
            raise ValueError("Integer n_components must be positive.")
        return min(requested, x_values.shape[0], x_values.shape[1])

    def fit(self, healthy_x: np.ndarray) -> "PCAReconstructionAnomalyDetector":
        x = np.asarray(healthy_x, dtype=float)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("healthy_x must be a non-empty 2D array.")
        self.pca_ = PCA(n_components=self._resolved_components(x))
        transformed = self.pca_.fit_transform(x)
        reconstructed = self.pca_.inverse_transform(transformed)
        errors = np.mean(np.square(x - reconstructed), axis=1)
        self.threshold_ = float(np.percentile(errors, self.threshold_percentile))
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_.tolist()
        return self

    def score(self, x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.pca_ is None:
            raise RuntimeError("Detector must be fitted before scoring.")
        x = np.asarray(x_values, dtype=float)
        transformed = self.pca_.transform(x)
        reconstructed = self.pca_.inverse_transform(transformed)
        errors = np.mean(np.square(x - reconstructed), axis=1)
        scores = errors.copy()
        flags = scores > self.threshold_
        return errors, scores, flags
