"""Operating-condition-aware feature normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from aeroguard.data.columns import OPERATIONAL_SETTING_COLUMNS
from aeroguard.data.operating_regimes import OperatingRegimeModel


def _safe_std(values: pd.Series) -> float:
    std = float(values.std(ddof=0))
    return std if np.isfinite(std) and std > 1.0e-12 else 1.0


@dataclass
class ConditionNormalizer:
    """Fit/apply bounded operating-condition normalization methods."""

    method: str = "none"
    feature_columns: list[str] | None = None
    n_regimes: int = 6
    random_state: int = 42
    ridge_alpha: float = 1.0
    healthy_column: str = "proxy_degradation_label"

    def __post_init__(self) -> None:
        if self.method not in {"none", "global_standardization", "regime_standardization", "residualization"}:
            raise ValueError("Unsupported condition-normalization method.")
        self.feature_columns_ = list(self.feature_columns or [])
        self.output_features_: list[str] = []
        self.global_means_: dict[str, float] = {}
        self.global_stds_: dict[str, float] = {}
        self.regime_model_: OperatingRegimeModel | None = None
        self.regime_stats_: dict[int, dict[str, tuple[float, float]]] = {}
        self.residual_models_: dict[str, Ridge] = {}
        self.residual_intercepts_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame, feature_columns: Iterable[str] | None = None) -> "ConditionNormalizer":
        self.feature_columns_ = list(feature_columns or self.feature_columns_)
        if not self.feature_columns_:
            raise ValueError("feature_columns must not be empty.")
        missing = [column for column in self.feature_columns_ if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing feature columns for normalization: {missing}")
        healthy = frame[frame[self.healthy_column] == 0] if self.healthy_column in frame.columns else frame
        if healthy.empty:
            healthy = frame
        for column in self.feature_columns_:
            self.global_means_[column] = float(healthy[column].mean())
            self.global_stds_[column] = _safe_std(healthy[column])
        if self.method == "regime_standardization":
            self.regime_model_ = OperatingRegimeModel(self.n_regimes, self.random_state).fit(frame)
            assigned = frame.copy()
            assigned["operating_regime"] = self.regime_model_.predict(assigned)
            healthy_assigned = assigned[assigned[self.healthy_column] == 0] if self.healthy_column in assigned.columns else assigned
            for regime, group in healthy_assigned.groupby("operating_regime"):
                self.regime_stats_[int(regime)] = {
                    column: (float(group[column].mean()), _safe_std(group[column]))
                    for column in self.feature_columns_
                }
        elif self.method == "residualization":
            x = healthy[OPERATIONAL_SETTING_COLUMNS].to_numpy(dtype=float)
            for column in self.feature_columns_:
                model = Ridge(alpha=float(self.ridge_alpha)).fit(x, healthy[column].to_numpy(dtype=float))
                self.residual_models_[column] = model
                residual = healthy[column].to_numpy(dtype=float) - model.predict(x)
                self.residual_intercepts_[column] = float(np.median(residual))
        self.output_features_ = self._output_names()
        return self

    def _output_names(self) -> list[str]:
        if self.method == "none":
            return list(self.feature_columns_)
        suffix = {
            "global_standardization": "_global_z",
            "regime_standardization": "_regime_z",
            "residualization": "_condition_residual",
        }[self.method]
        return [f"{column}{suffix}" for column in self.feature_columns_]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.output_features_:
            raise RuntimeError("ConditionNormalizer must be fitted before transform.")
        result = frame.copy()
        if self.method == "none":
            return result
        if self.method == "global_standardization":
            for column in self.feature_columns_:
                result[f"{column}_global_z"] = (result[column] - self.global_means_[column]) / self.global_stds_[column]
        elif self.method == "regime_standardization":
            if self.regime_model_ is None:
                raise RuntimeError("Regime model was not fitted.")
            result["operating_regime"] = self.regime_model_.predict(result)
            for column in self.feature_columns_:
                mean_map = {regime: stats[column][0] for regime, stats in self.regime_stats_.items() if column in stats}
                std_map = {regime: stats[column][1] for regime, stats in self.regime_stats_.items() if column in stats}
                means = result["operating_regime"].map(mean_map).fillna(self.global_means_[column]).to_numpy(dtype=float)
                stds = result["operating_regime"].map(std_map).fillna(self.global_stds_[column]).to_numpy(dtype=float)
                result[f"{column}_regime_z"] = (result[column].to_numpy(dtype=float) - means) / stds
        else:
            x = result[OPERATIONAL_SETTING_COLUMNS].to_numpy(dtype=float)
            for column in self.feature_columns_:
                prediction = self.residual_models_[column].predict(x)
                result[f"{column}_condition_residual"] = result[column].to_numpy(dtype=float) - prediction
        return result

    def fit_transform(self, frame: pd.DataFrame, feature_columns: Iterable[str] | None = None) -> pd.DataFrame:
        self.fit(frame, feature_columns=feature_columns)
        return self.transform(frame)

    def metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "method": self.method,
            "feature_columns": self.feature_columns_,
            "output_features": self.output_features_,
        }
        if self.regime_model_ is not None:
            metadata["regime_model"] = self.regime_model_.metadata()
        if self.method == "residualization":
            metadata["ridge_alpha"] = float(self.ridge_alpha)
        return metadata
