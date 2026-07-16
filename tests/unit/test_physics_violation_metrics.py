import pytest

from aeroguard.deep.physics.violation_metrics import (
    cycle_rate_metrics,
    health_consistency_metrics,
    monotonicity_metrics,
    optimistic_error_metrics,
    regime_consistency_metrics,
    smoothness_metrics,
)


def test_perfectly_monotonic_and_violating_sequence() -> None:
    perfect = monotonicity_metrics([10, 9], [9, 8])
    violating = monotonicity_metrics([10, 9], [11, 8])

    assert perfect["violation_rate"] == 0.0
    assert violating["violation_count"] == 1.0


def test_perfect_rate_consistency_and_smooth_trajectory() -> None:
    assert cycle_rate_metrics([10, 8], [9, 6], [1, 2])["rate_rmse"] == 0.0
    assert smoothness_metrics([10], [8], [6])["smoothness_violation_rate"] == 0.0


def test_health_and_regime_metrics() -> None:
    health = health_consistency_metrics([10, 8, 6], [0.9, 0.7, 0.5], [0], [1])
    regime = regime_consistency_metrics([0.1, 0.2], [1.0, 4.0], tolerance=2.0)

    assert health["health_rul_spearman"] > 0.9
    assert regime["regime_consistency_violation_rate"] == 0.5


def test_optimistic_and_low_rul_metrics() -> None:
    metrics = optimistic_error_metrics([5, 50], [10, 40], severe_threshold=3, low_rul_threshold=10)

    assert metrics["optimistic_prediction_rate"] == 0.5
    assert metrics["severe_optimistic_prediction_rate"] == 0.5
    assert metrics["low_rul_optimistic_error_rate"] == 1.0


def test_empty_input_handling_and_nonfinite_rejection() -> None:
    assert monotonicity_metrics([], [])["violation_rate"] == 0.0
    with pytest.raises(ValueError, match="finite"):
        monotonicity_metrics([1.0], [float("nan")])
