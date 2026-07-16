"""Validation-safe anomaly score calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoreCalibrator:
    """Calibrate detector scores using healthy model-training scores only."""

    method: str = "empirical_percentile"
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    epsilon: float = 1e-9
    clip: bool = True

    def __post_init__(self) -> None:
        if self.method not in {"empirical_percentile", "robust_z", "quantile"}:
            raise ValueError("Unsupported calibration method.")
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("Calibration quantiles must satisfy 0 <= low < high <= 1.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        self.sorted_scores_: np.ndarray | None = None
        self.median_: float | None = None
        self.mad_: float | None = None
        self.lower_: float | None = None
        self.upper_: float | None = None
        self.scale_: float | None = None

    def fit(self, healthy_scores: object) -> "ScoreCalibrator":
        scores = np.asarray(healthy_scores, dtype=float)
        if scores.ndim != 1 or len(scores) == 0:
            raise ValueError("healthy_scores must be a non-empty one-dimensional array.")
        if not np.isfinite(scores).all():
            raise ValueError("healthy_scores must be finite.")
        self.sorted_scores_ = np.sort(scores)
        self.median_ = float(np.median(scores))
        self.mad_ = float(np.median(np.abs(scores - self.median_)))
        self.lower_ = float(np.quantile(scores, self.lower_quantile))
        self.upper_ = float(np.quantile(scores, self.upper_quantile))
        robust_scale = self.mad_ * 1.4826
        self.scale_ = float(max(robust_scale, self.epsilon))
        if self.upper_ <= self.lower_:
            self.upper_ = self.lower_ + self.epsilon
        return self

    def transform(self, scores: object) -> np.ndarray:
        if self.sorted_scores_ is None:
            raise RuntimeError("ScoreCalibrator must be fitted before transform.")
        values = np.asarray(scores, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("scores must be finite.")
        if self.method == "empirical_percentile":
            ranks = np.searchsorted(self.sorted_scores_, values, side="right")
            calibrated = ranks / len(self.sorted_scores_)
        elif self.method == "robust_z":
            calibrated = (values - self.median_) / self.scale_
        else:
            calibrated = (values - self.lower_) / (self.upper_ - self.lower_)
        if self.clip and self.method in {"empirical_percentile", "quantile"}:
            calibrated = np.clip(calibrated, 0.0, 1.0)
        return calibrated.astype(float)

    def fit_transform(self, healthy_scores: object, scores: object) -> np.ndarray:
        self.fit(healthy_scores)
        return self.transform(scores)

    def metadata(self) -> dict[str, float | str | bool | int]:
        if self.sorted_scores_ is None:
            raise RuntimeError("ScoreCalibrator has no metadata before fit.")
        return {
            "method": self.method,
            "healthy_score_count": int(len(self.sorted_scores_)),
            "median": self.median_,
            "mad": self.mad_,
            "robust_scale": self.scale_,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "lower_value": self.lower_,
            "upper_value": self.upper_,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "score_direction": "higher calibrated score means more anomalous",
        }


def calibrate_detector_scores(
    healthy_scores_by_detector: dict[str, object],
    scores_by_detector: dict[str, object],
    method: str,
    lower_quantile: float,
    upper_quantile: float,
    epsilon: float,
    clip: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Fit one calibrator per detector and transform score arrays."""
    calibrated: dict[str, np.ndarray] = {}
    metadata: dict[str, dict] = {}
    for detector, healthy_scores in healthy_scores_by_detector.items():
        calibrator = ScoreCalibrator(
            method=method,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            epsilon=epsilon,
            clip=clip,
        ).fit(healthy_scores)
        calibrated[detector] = calibrator.transform(scores_by_detector[detector])
        metadata[detector] = calibrator.metadata()
    return calibrated, metadata
