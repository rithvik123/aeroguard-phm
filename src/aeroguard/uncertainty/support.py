"""Training-support and out-of-distribution scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def _safe_scale(values: pd.Series) -> float:
    q75 = float(values.quantile(0.75))
    q25 = float(values.quantile(0.25))
    scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = float(values.std(ddof=0))
    return scale if np.isfinite(scale) and scale > 1.0e-12 else 1.0


@dataclass
class SupportModel:
    feature_columns: list[str]
    percentile_low: float = 0.01
    percentile_high: float = 0.99
    feature_exceedance_limited: float = 0.05
    feature_exceedance_out: float = 0.20
    robust_distance_limited: float = 3.0
    robust_distance_out: float = 6.0
    regime_distance_quantile: float = 0.99

    def fit(self, frame: pd.DataFrame) -> "SupportModel":
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing support features: {missing}")
        values = frame[self.feature_columns].astype(float)
        self.low_: dict[str, float] = values.quantile(self.percentile_low).to_dict()
        self.high_: dict[str, float] = values.quantile(self.percentile_high).to_dict()
        self.median_: dict[str, float] = values.median().to_dict()
        self.scale_: dict[str, float] = {column: _safe_scale(values[column]) for column in self.feature_columns}
        if "operating_regime" in frame.columns:
            centers = frame.groupby("operating_regime")[self.feature_columns].median()
            self.regime_centers_ = centers
            train_dist = self._nearest_regime_distance(values)
            self.regime_distance_threshold_ = float(np.quantile(train_dist, self.regime_distance_quantile))
        else:
            self.regime_centers_ = pd.DataFrame(columns=self.feature_columns)
            self.regime_distance_threshold_ = float("inf")
        return self

    def _nearest_regime_distance(self, values: pd.DataFrame) -> np.ndarray:
        if self.regime_centers_.empty:
            return np.zeros(len(values), dtype=float)
        scales = np.asarray([self.scale_[column] for column in self.feature_columns], dtype=float)
        x = values[self.feature_columns].to_numpy(dtype=float)
        centers = self.regime_centers_[self.feature_columns].to_numpy(dtype=float)
        distances = []
        for center in centers:
            distances.append(np.sqrt(np.mean(np.square((x - center) / scales), axis=1)))
        return np.min(np.column_stack(distances), axis=1)

    def score(self, frame: pd.DataFrame, interval_width_ratio: Iterable[float] | None = None) -> pd.DataFrame:
        if not hasattr(self, "low_"):
            raise RuntimeError("SupportModel must be fitted before scoring.")
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing support features: {missing}")
        values = frame[self.feature_columns].astype(float)
        finite = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
        outside = np.zeros((len(values), len(self.feature_columns)), dtype=bool)
        for idx, column in enumerate(self.feature_columns):
            outside[:, idx] = (values[column] < self.low_[column]) | (values[column] > self.high_[column])
        exceedance = outside.mean(axis=1)
        z = np.column_stack(
            [
                np.abs((values[column].to_numpy(dtype=float) - self.median_[column]) / self.scale_[column])
                for column in self.feature_columns
            ]
        )
        robust_distance = np.sqrt(np.mean(np.square(z), axis=1))
        regime_distance = self._nearest_regime_distance(values)
        width_ratio = np.asarray(list(interval_width_ratio), dtype=float) if interval_width_ratio is not None else np.ones(len(values))
        score = (
            0.45 * np.clip(exceedance / max(self.feature_exceedance_out, 1.0e-9), 0, 1)
            + 0.35 * np.clip(robust_distance / max(self.robust_distance_out, 1.0e-9), 0, 1)
            + 0.20 * np.clip(regime_distance / max(self.regime_distance_threshold_, 1.0e-9), 0, 1)
        )
        status = np.full(len(values), "IN_SUPPORT", dtype=object)
        limited = (
            (exceedance > self.feature_exceedance_limited)
            | (robust_distance > self.robust_distance_limited)
            | (regime_distance > self.regime_distance_threshold_)
        )
        out = (
            (exceedance > self.feature_exceedance_out)
            | (robust_distance > self.robust_distance_out)
            | ~finite
        )
        status[limited] = "LIMITED_SUPPORT"
        status[out] = "OUT_OF_SUPPORT"
        return pd.DataFrame(
            {
                "support_status": status,
                "support_score": score,
                "feature_exceedance_fraction": exceedance,
                "robust_distance_score": robust_distance,
                "regime_distance": regime_distance,
                "regime_distance_threshold": self.regime_distance_threshold_,
                "interval_width_ratio": width_ratio,
                "non_finite_input": ~finite,
            },
            index=frame.index,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "feature_count": len(self.feature_columns),
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "feature_exceedance_limited": self.feature_exceedance_limited,
            "feature_exceedance_out": self.feature_exceedance_out,
            "robust_distance_limited": self.robust_distance_limited,
            "robust_distance_out": self.robust_distance_out,
            "regime_distance_threshold": self.regime_distance_threshold_,
        }
