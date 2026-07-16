"""Grouped conformal interval helpers for RUL prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def validate_nominal_level(level: float) -> float:
    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError("Nominal coverage level must be in (0, 1).")
    return level


def conformal_quantile(scores: Iterable[float], nominal_level: float) -> float:
    """Finite-sample conformal quantile for absolute residual scores."""
    level = validate_nominal_level(nominal_level)
    values = np.asarray(list(scores), dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Conformal scores must be a non-empty one-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError("Conformal scores must be finite.")
    sorted_scores = np.sort(values)
    rank = int(np.ceil((len(sorted_scores) + 1) * level))
    rank = min(max(rank, 1), len(sorted_scores))
    return float(sorted_scores[rank - 1])


def symmetric_intervals(predicted: Iterable[float], radius: float, clip_lower: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(predicted, dtype=float)
    if pred.ndim != 1 or not np.isfinite(pred).all():
        raise ValueError("Predictions must be a finite one-dimensional array.")
    raw_lower = pred - float(radius)
    lower = np.maximum(0.0, raw_lower) if clip_lower else raw_lower
    upper = pred + float(radius)
    return lower.astype(float), upper.astype(float), raw_lower.astype(float)


def assign_predicted_rul_band(values: Iterable[float], bands: list[dict[str, float | str]]) -> list[str]:
    result = []
    for value in np.asarray(list(values), dtype=float):
        label = None
        for band in bands:
            lower = float(band["lower"])
            upper = band.get("upper")
            upper_value = np.inf if upper is None else float(upper)
            if lower <= value <= upper_value:
                label = str(band["label"])
                break
        if label is None:
            label = str(bands[-1]["label"])
        result.append(label)
    return result


@dataclass
class GlobalConformalCalibrator:
    nominal_levels: list[float]

    def fit(self, residuals: Iterable[float]) -> "GlobalConformalCalibrator":
        values = np.abs(np.asarray(list(residuals), dtype=float))
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("Residuals must be non-empty and finite.")
        self.sample_count_ = int(len(values))
        self.quantiles_ = {float(level): conformal_quantile(values, float(level)) for level in self.nominal_levels}
        return self

    def interval_frame(self, predicted: Iterable[float], prefix: str = "") -> pd.DataFrame:
        if not hasattr(self, "quantiles_"):
            raise RuntimeError("GlobalConformalCalibrator must be fitted before interval generation.")
        pred = np.asarray(list(predicted), dtype=float)
        output: dict[str, np.ndarray] = {}
        for level, radius in self.quantiles_.items():
            pct = int(round(level * 100))
            lower, upper, raw_lower = symmetric_intervals(pred, radius)
            output[f"{prefix}lower_{pct}"] = lower
            output[f"{prefix}upper_{pct}"] = upper
            output[f"{prefix}raw_lower_{pct}"] = raw_lower
            output[f"{prefix}radius_{pct}"] = np.repeat(radius, len(pred))
        return pd.DataFrame(output)

    def metadata(self) -> dict[str, object]:
        return {"sample_count": self.sample_count_, "quantiles": {str(k): v for k, v in self.quantiles_.items()}}


@dataclass
class PredictedRulBandConformalCalibrator:
    nominal_levels: list[float]
    bands: list[dict[str, float | str]]
    minimum_samples_per_band: int = 20

    def fit(self, predicted: Iterable[float], residuals: Iterable[float]) -> "PredictedRulBandConformalCalibrator":
        pred = np.asarray(list(predicted), dtype=float)
        values = np.abs(np.asarray(list(residuals), dtype=float))
        if len(pred) == 0 or len(pred) != len(values):
            raise ValueError("Predictions and residuals must be non-empty and aligned.")
        if not np.isfinite(pred).all() or not np.isfinite(values).all():
            raise ValueError("Predictions and residuals must be finite.")
        labels = np.asarray(assign_predicted_rul_band(pred, self.bands), dtype=object)
        self.global_ = GlobalConformalCalibrator(self.nominal_levels).fit(values)
        self.band_quantiles_: dict[str, dict[float, float]] = {}
        self.band_counts_: dict[str, int] = {}
        self.band_fallback_: dict[str, bool] = {}
        for band in self.bands:
            label = str(band["label"])
            band_values = values[labels == label]
            self.band_counts_[label] = int(len(band_values))
            use_fallback = len(band_values) < int(self.minimum_samples_per_band)
            self.band_fallback_[label] = bool(use_fallback)
            source = values if use_fallback else band_values
            self.band_quantiles_[label] = {
                float(level): conformal_quantile(source, float(level)) for level in self.nominal_levels
            }
        return self

    def interval_frame(self, predicted: Iterable[float], prefix: str = "") -> pd.DataFrame:
        if not hasattr(self, "band_quantiles_"):
            raise RuntimeError("PredictedRulBandConformalCalibrator must be fitted first.")
        pred = np.asarray(list(predicted), dtype=float)
        labels = assign_predicted_rul_band(pred, self.bands)
        output: dict[str, list[float | str | bool]] = {
            f"{prefix}predicted_rul_band": labels,
            f"{prefix}band_fallback_used": [self.band_fallback_.get(label, True) for label in labels],
        }
        for level in self.nominal_levels:
            pct = int(round(level * 100))
            lowers, uppers, raw_lowers, radii = [], [], [], []
            for value, label in zip(pred, labels):
                radius = self.band_quantiles_[label][float(level)]
                lower, upper, raw_lower = symmetric_intervals([value], radius)
                lowers.append(float(lower[0]))
                uppers.append(float(upper[0]))
                raw_lowers.append(float(raw_lower[0]))
                radii.append(float(radius))
            output[f"{prefix}lower_{pct}"] = lowers
            output[f"{prefix}upper_{pct}"] = uppers
            output[f"{prefix}raw_lower_{pct}"] = raw_lowers
            output[f"{prefix}radius_{pct}"] = radii
        return pd.DataFrame(output)

    def metadata(self) -> dict[str, object]:
        return {
            "minimum_samples_per_band": int(self.minimum_samples_per_band),
            "global": self.global_.metadata(),
            "bands": [
                {
                    "label": label,
                    "sample_count": self.band_counts_[label],
                    "fallback_used": self.band_fallback_[label],
                    "quantiles": {str(k): v for k, v in self.band_quantiles_[label].items()},
                }
                for label in self.band_quantiles_
            ],
        }


def conformalize_interval(
    true_values: Iterable[float],
    lower: Iterable[float],
    upper: Iterable[float],
    nominal_levels: list[float],
) -> dict[float, float]:
    true = np.asarray(list(true_values), dtype=float)
    lo = np.asarray(list(lower), dtype=float)
    hi = np.asarray(list(upper), dtype=float)
    if len(true) == 0 or not (len(true) == len(lo) == len(hi)):
        raise ValueError("Conformalized interval inputs must be non-empty and aligned.")
    if not np.isfinite(true).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("Conformalized interval inputs must be finite.")
    scores = np.maximum(lo - true, true - hi)
    scores = np.maximum(scores, 0.0)
    return {float(level): conformal_quantile(scores, float(level)) for level in nominal_levels}
