"""Operating-condition regime modelling from C-MAPSS settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from aeroguard.data.columns import OPERATIONAL_SETTING_COLUMNS


@dataclass
class OperatingRegimeModel:
    """Small deterministic clustering model over operational settings."""

    n_regimes: int = 6
    random_state: int = 42

    def __post_init__(self) -> None:
        if int(self.n_regimes) <= 0:
            raise ValueError("n_regimes must be positive.")
        self.scaler_: StandardScaler | None = None
        self.model_: KMeans | None = None
        self.columns_: list[str] = list(OPERATIONAL_SETTING_COLUMNS)

    def fit(self, frame: pd.DataFrame, columns: Iterable[str] | None = None) -> "OperatingRegimeModel":
        self.columns_ = list(columns or OPERATIONAL_SETTING_COLUMNS)
        if not set(self.columns_).issubset(frame.columns):
            raise ValueError("Missing operational-setting columns for regime fitting.")
        x = frame[self.columns_].to_numpy(dtype=float)
        if len(x) == 0:
            raise ValueError("Cannot fit regimes on an empty frame.")
        n_clusters = min(int(self.n_regimes), len(np.unique(x, axis=0)))
        n_clusters = max(1, n_clusters)
        self.scaler_ = StandardScaler().fit(x)
        scaled = self.scaler_.transform(x)
        self.model_ = KMeans(n_clusters=n_clusters, random_state=int(self.random_state), n_init=10)
        self.model_.fit(scaled)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.scaler_ is None or self.model_ is None:
            raise RuntimeError("OperatingRegimeModel must be fitted before predict.")
        x = frame[self.columns_].to_numpy(dtype=float)
        return self.model_.predict(self.scaler_.transform(x)).astype(int)

    def fit_predict(self, frame: pd.DataFrame, columns: Iterable[str] | None = None) -> np.ndarray:
        self.fit(frame, columns=columns)
        return self.predict(frame)

    def metadata(self) -> dict[str, object]:
        if self.model_ is None:
            raise RuntimeError("OperatingRegimeModel has no metadata before fit.")
        return {
            "n_regimes_requested": int(self.n_regimes),
            "n_regimes_fitted": int(self.model_.n_clusters),
            "random_state": int(self.random_state),
            "columns": self.columns_,
        }


def assign_operating_regimes(
    frame: pd.DataFrame,
    model: OperatingRegimeModel,
    output_column: str = "operating_regime",
) -> pd.DataFrame:
    result = frame.copy()
    result[output_column] = model.predict(result)
    return result


def regime_counts(frame: pd.DataFrame, regime_column: str = "operating_regime") -> dict[str, int]:
    if regime_column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[regime_column].value_counts().sort_index().items()}
