"""Leakage-aware feature selection, audit, and scaling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from aeroguard.data.columns import (
    BASE_FEATURE_COLUMNS,
    CYCLE_COLUMN,
    EXCLUDED_MODEL_INPUT_COLUMNS,
    SENSOR_COLUMNS,
    UNIT_COLUMN,
)


@dataclass
class AeroGuardPreprocessor:
    """Feature selector and StandardScaler fitted only on training engines."""

    include_cycle_as_feature: bool = False
    features_to_exclude: Iterable[str] = field(default_factory=list)
    near_constant_threshold: float = 1e-10
    rolling_features_enabled: bool = False
    rolling_window_sizes: Iterable[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.near_constant_threshold < 0:
            raise ValueError("near_constant_threshold must be non-negative.")
        self.features_to_exclude = list(self.features_to_exclude)
        self.rolling_window_sizes = [int(size) for size in self.rolling_window_sizes]
        if any(size <= 1 for size in self.rolling_window_sizes):
            raise ValueError("rolling_window_sizes must contain integers greater than 1.")
        self.scaler_: StandardScaler | None = None
        self.retained_feature_names_: list[str] = []
        self.candidate_feature_names_: list[str] = []
        self.feature_exclusion_reasons_: dict[str, str] = {}
        self.feature_audit_: pd.DataFrame | None = None
        self.correlation_audit_: pd.DataFrame | None = None

    @property
    def retained_feature_names(self) -> list[str]:
        """Final feature names retained after fit-time selection."""
        return list(self.retained_feature_names_)

    def _candidate_features(self, frame: pd.DataFrame) -> list[str]:
        features = list(BASE_FEATURE_COLUMNS)
        if self.include_cycle_as_feature:
            features.insert(0, CYCLE_COLUMN)
        rolling = [column for column in frame.columns if column.startswith("rolling_")]
        return [column for column in [*features, *rolling] if column in frame.columns]

    def _with_optional_rolling(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.rolling_features_enabled:
            return frame.copy()
        result = frame.copy()
        sorted_frame = result.sort_values([UNIT_COLUMN, CYCLE_COLUMN])
        for window in self.rolling_window_sizes:
            for sensor in SENSOR_COLUMNS:
                if sensor not in sorted_frame.columns:
                    continue
                rolling_name = f"rolling_mean_w{window}_{sensor}"
                rolled = (
                    sorted_frame.groupby(UNIT_COLUMN, sort=False)[sensor]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                result.loc[sorted_frame.index, rolling_name] = rolled.to_numpy()
        return result

    def fit(self, frame: pd.DataFrame, correlation_threshold: float = 0.95) -> "AeroGuardPreprocessor":
        """Fit feature exclusions and scaling from the model-training engine split."""
        prepared = self._with_optional_rolling(frame)
        candidates = self._candidate_features(prepared)
        configured_exclusions = set(self.features_to_exclude)
        forbidden = set(EXCLUDED_MODEL_INPUT_COLUMNS)
        reasons: dict[str, str] = {}

        for feature in candidates:
            if feature in configured_exclusions:
                reasons[feature] = "configured exclusion"
            elif feature in forbidden:
                reasons[feature] = "identifier, cycle, target, or prediction column"
            else:
                series = pd.to_numeric(prepared[feature], errors="coerce")
                unique_count = int(series.nunique(dropna=True))
                variance = float(series.var(ddof=0)) if len(series) else np.nan
                if unique_count <= 1:
                    reasons[feature] = "constant in model-training engines"
                elif np.isfinite(variance) and variance <= self.near_constant_threshold:
                    reasons[feature] = (
                        "near-constant in model-training engines "
                        f"(variance <= {self.near_constant_threshold})"
                    )

        retained = [feature for feature in candidates if feature not in reasons]
        if not retained:
            raise ValueError("No features retained after fit-time feature selection.")

        self.candidate_feature_names_ = candidates
        self.feature_exclusion_reasons_ = reasons
        self.retained_feature_names_ = retained
        self.feature_audit_, self.correlation_audit_ = audit_features(
            prepared,
            candidate_features=candidates,
            retained_features=retained,
            exclusion_reasons=reasons,
            near_constant_threshold=self.near_constant_threshold,
            correlation_threshold=correlation_threshold,
        )
        self.scaler_ = StandardScaler()
        self.scaler_.fit(prepared[retained])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply retained features and fitted scaler to a new split."""
        if self.scaler_ is None:
            raise RuntimeError("AeroGuardPreprocessor must be fitted before transform.")
        prepared = self._with_optional_rolling(frame)
        missing = [name for name in self.retained_feature_names_ if name not in prepared]
        if missing:
            raise ValueError(f"Input frame is missing retained features: {missing}")
        return self.scaler_.transform(prepared[self.retained_feature_names_])

    def fit_transform(
        self,
        frame: pd.DataFrame,
        correlation_threshold: float = 0.95,
    ) -> np.ndarray:
        """Fit on a training split and return scaled training features."""
        return self.fit(frame, correlation_threshold=correlation_threshold).transform(frame)


def audit_features(
    frame: pd.DataFrame,
    candidate_features: Iterable[str],
    retained_features: Iterable[str],
    exclusion_reasons: dict[str, str],
    near_constant_threshold: float,
    correlation_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create feature and correlation audit tables from training-engine data."""
    retained = set(retained_features)
    rows: list[dict[str, object]] = []
    for feature in candidate_features:
        numeric = pd.to_numeric(frame[feature], errors="coerce")
        finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
        variance = float(finite.var(ddof=0)) if len(finite) else np.nan
        unique_count = int(finite.nunique(dropna=True))
        is_constant = unique_count <= 1
        is_near_constant = (
            bool(np.isfinite(variance) and variance <= near_constant_threshold)
            and not is_constant
        )
        rows.append(
            {
                "feature": feature,
                "dtype": str(frame[feature].dtype),
                "missing_count": int(numeric.isna().sum()),
                "infinite_count": int(np.isinf(numeric.to_numpy(dtype=float)).sum()),
                "unique_count": unique_count,
                "minimum": float(finite.min()) if len(finite) else np.nan,
                "maximum": float(finite.max()) if len(finite) else np.nan,
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "standard_deviation": float(finite.std(ddof=0)) if len(finite) else np.nan,
                "variance": variance,
                "is_constant": is_constant,
                "is_near_constant": is_near_constant,
                "selected_for_model": feature in retained,
                "exclusion_reason": exclusion_reasons.get(feature, ""),
            }
        )

    audit = pd.DataFrame(rows)
    usable = [feature for feature in candidate_features if feature in frame.columns]
    corr_rows: list[dict[str, object]] = []
    if len(usable) >= 2:
        corr = frame[usable].corr(numeric_only=True).abs()
        for left_idx, left in enumerate(usable):
            for right in usable[left_idx + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value) and value >= correlation_threshold:
                    corr_rows.append(
                        {
                            "feature_1": left,
                            "feature_2": right,
                            "absolute_correlation": float(value),
                            "threshold": float(correlation_threshold),
                            "action": "recorded only; not automatically removed",
                        }
                    )
    correlation_audit = pd.DataFrame(
        corr_rows,
        columns=[
            "feature_1",
            "feature_2",
            "absolute_correlation",
            "threshold",
            "action",
        ],
    )
    return audit, correlation_audit
