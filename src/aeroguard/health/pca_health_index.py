"""Transparent PCA-based health-index estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class PCAHealthIndex:
    """Construct a scalar health index from the first PCA component.

    The PCA model is fitted on model-training rows only. The first component is
    sign-oriented using only model-training uncapped RUL so that larger values
    indicate healthier states. Robust quantiles from the model-training rows
    scale the oriented index to an approximately 0-1 presentation range.
    """

    n_components: int | float = 0.95
    lower_quantile: float = 0.02
    upper_quantile: float = 0.98
    clip_scaled: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError("Health-index quantiles must satisfy 0 <= low < high <= 1.")
        self.pca_: PCA | None = None
        self.orientation_: float = 1.0
        self.scale_low_: float = 0.0
        self.scale_high_: float = 1.0
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

    def fit(self, x_values: np.ndarray, rul_uncapped: np.ndarray) -> "PCAHealthIndex":
        x = np.asarray(x_values, dtype=float)
        rul = np.asarray(rul_uncapped, dtype=float)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("x_values must be a non-empty 2D array.")
        if len(rul) != len(x):
            raise ValueError("rul_uncapped length must match x_values rows.")
        if not np.isfinite(x).all() or not np.isfinite(rul).all():
            raise ValueError("PCA health-index inputs must be finite.")

        self.pca_ = PCA(n_components=self._resolved_components(x))
        scores = self.pca_.fit_transform(x)
        first_component = scores[:, 0]
        corr = np.corrcoef(first_component, rul)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        self.orientation_ = 1.0 if corr >= 0 else -1.0
        oriented = first_component * self.orientation_
        self.scale_low_ = float(np.quantile(oriented, self.lower_quantile))
        self.scale_high_ = float(np.quantile(oriented, self.upper_quantile))
        if self.scale_high_ <= self.scale_low_:
            self.scale_low_ = float(oriented.min())
            self.scale_high_ = float(oriented.max())
        if self.scale_high_ <= self.scale_low_:
            self.scale_high_ = self.scale_low_ + 1.0
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_.tolist()
        return self

    def transform(self, x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.pca_ is None:
            raise RuntimeError("PCAHealthIndex must be fitted before transform.")
        x = np.asarray(x_values, dtype=float)
        scores = self.pca_.transform(x)
        raw = scores[:, 0] * self.orientation_
        scaled = (raw - self.scale_low_) / (self.scale_high_ - self.scale_low_)
        if self.clip_scaled:
            scaled = np.clip(scaled, 0.0, 1.0)
        return raw, scaled

    def fit_transform(
        self,
        x_values: np.ndarray,
        rul_uncapped: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.fit(x_values, rul_uncapped)
        return self.transform(x_values)
