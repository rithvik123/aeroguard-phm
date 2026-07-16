import numpy as np

from aeroguard.anomaly.pca_reconstruction import PCAReconstructionAnomalyDetector


def test_pca_reconstruction_scores_large_outlier_as_anomaly() -> None:
    rng = np.random.default_rng(1)
    healthy = rng.normal(0, 0.1, size=(80, 3))
    detector = PCAReconstructionAnomalyDetector(n_components=1, threshold_percentile=95).fit(healthy)

    _, scores, flags = detector.score(np.vstack([healthy[:3], np.array([[5.0, -5.0, 5.0]])]))

    assert scores[-1] > scores[:3].max()
    assert flags[-1]
    assert detector.threshold_ >= 0


def test_pca_reconstruction_requires_fit_before_score() -> None:
    detector = PCAReconstructionAnomalyDetector(n_components=1)

    try:
        detector.score(np.ones((2, 2)))
    except RuntimeError as exc:
        assert "fitted" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")
