import pytest

from aeroguard.evaluation.uncertainty_metrics import interval_metrics


def test_interval_metrics_perfect_coverage_and_width() -> None:
    metrics = interval_metrics([10, 20], [10, 20], [9, 19], [11, 21], 0.9)

    assert metrics["coverage"] == 1.0
    assert metrics["overcoverage_amount"] == pytest.approx(0.1)
    assert metrics["mean_interval_width"] == 2.0


def test_interval_metrics_undercoverage_and_crossing() -> None:
    metrics = interval_metrics([10, 20], [10, 20], [11, 21], [12, 22], 0.9)

    assert metrics["coverage"] == 0.0
    assert metrics["undercoverage_amount"] == 0.9
    assert metrics["lower_bound_violation_rate"] == 1.0


def test_interval_metrics_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        interval_metrics([], [], [], [], 0.9)
    with pytest.raises(ValueError):
        interval_metrics([1], [1], [0], [2], 1.1)
