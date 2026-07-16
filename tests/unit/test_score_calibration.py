import numpy as np

from aeroguard.anomaly.score_calibration import ScoreCalibrator


def test_empirical_percentile_score_direction() -> None:
    calibrator = ScoreCalibrator(method="empirical_percentile").fit([1, 2, 3, 4])

    scores = calibrator.transform([0, 2, 4, 5])

    assert scores.tolist() == [0.0, 0.5, 1.0, 1.0]
    assert scores[-1] >= scores[-2] >= scores[1] >= scores[0]


def test_robust_z_handles_zero_mad() -> None:
    calibrator = ScoreCalibrator(method="robust_z", epsilon=1e-6).fit([2, 2, 2])

    scores = calibrator.transform([2, 3])

    assert np.isfinite(scores).all()
    assert scores[1] > scores[0]


def test_quantile_scaling_clips_for_presentation() -> None:
    calibrator = ScoreCalibrator(method="quantile", lower_quantile=0.25, upper_quantile=0.75, clip=True).fit([0, 1, 2, 3, 4])

    scores = calibrator.transform([-1, 2, 10])

    assert scores[0] == 0.0
    assert 0.0 < scores[1] < 1.0
    assert scores[2] == 1.0
