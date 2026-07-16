import numpy as np
import pytest

from aeroguard.uncertainty.conformal import (
    GlobalConformalCalibrator,
    PredictedRulBandConformalCalibrator,
    assign_predicted_rul_band,
    conformal_quantile,
    symmetric_intervals,
)


def test_conformal_perfect_residuals_and_interval_ordering() -> None:
    calibrator = GlobalConformalCalibrator([0.8, 0.9, 0.95]).fit([0.0, 0.0, 0.0])
    intervals = calibrator.interval_frame([5.0, 10.0])

    assert intervals["lower_90"].tolist() == [5.0, 10.0]
    assert intervals["upper_90"].tolist() == [5.0, 10.0]
    assert (intervals["lower_95"] <= intervals["upper_95"]).all()


def test_finite_sample_correction_and_nonnegative_presentation_lower_bound() -> None:
    assert conformal_quantile([1.0, 2.0, 3.0, 4.0], 0.8) == 4.0
    lower, upper, raw_lower = symmetric_intervals([2.0], 5.0)

    assert lower[0] == 0.0
    assert upper[0] == 7.0
    assert raw_lower[0] == -3.0


def test_conformal_rejects_empty_and_invalid_alpha() -> None:
    with pytest.raises(ValueError):
        conformal_quantile([], 0.9)
    with pytest.raises(ValueError):
        conformal_quantile([1.0], 1.0)


def test_predicted_rul_band_conformal_fallback_and_boundaries() -> None:
    bands = [
        {"label": "low", "lower": 0, "upper": 10},
        {"label": "high", "lower": 10.000001, "upper": None},
    ]
    calibrator = PredictedRulBandConformalCalibrator([0.9], bands, minimum_samples_per_band=3)
    calibrator.fit([1.0, 2.0, 20.0, 30.0], [1.0, 2.0, 3.0, 4.0])
    intervals = calibrator.interval_frame([10.0, 11.0])

    assert assign_predicted_rul_band([10.0, 10.1], bands) == ["low", "high"]
    assert intervals["predicted_rul_band"].tolist() == ["low", "high"]
    assert intervals["band_fallback_used"].tolist() == [True, True]
    assert np.isfinite(intervals[["lower_90", "upper_90"]].to_numpy()).all()
