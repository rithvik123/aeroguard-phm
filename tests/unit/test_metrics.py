import math

import pytest

from aeroguard.evaluation.metrics import (
    mean_absolute_error,
    nasa_asymmetric_score,
    per_engine_prediction_frame,
    prediction_direction,
    r2_score,
    regression_metrics,
    root_mean_squared_error,
)


def test_perfect_predictions_score_zero() -> None:
    y_true = [10, 20, 30]
    y_pred = [10, 20, 30]

    assert mean_absolute_error(y_true, y_pred) == 0.0
    assert root_mean_squared_error(y_true, y_pred) == 0.0
    assert r2_score(y_true, y_pred) == 1.0
    assert nasa_asymmetric_score(y_true, y_pred) == 0.0


def test_late_or_optimistic_prediction_has_stronger_penalty() -> None:
    true = [100]
    conservative_prediction = [90]
    optimistic_prediction = [110]

    conservative_score = nasa_asymmetric_score(true, conservative_prediction)
    optimistic_score = nasa_asymmetric_score(true, optimistic_prediction)

    assert optimistic_score > conservative_score
    assert prediction_direction(-10) == "under_prediction_conservative"
    assert prediction_direction(10) == "over_prediction_optimistic"


def test_multiple_engine_aggregation_and_prediction_frame() -> None:
    metrics = regression_metrics([10, 20], [8, 25])
    frame = per_engine_prediction_frame([1, 2], [10, 20], [8, 25], "demo")

    assert metrics["mae"] == 3.5
    assert math.isclose(metrics["rmse"], math.sqrt((4 + 25) / 2))
    assert frame["absolute_error"].tolist() == [2.0, 5.0]
    assert frame["prediction_direction"].tolist() == [
        "under_prediction_conservative",
        "over_prediction_optimistic",
    ]


def test_metric_inputs_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        mean_absolute_error([], [])


def test_metric_inputs_must_have_matching_lengths() -> None:
    with pytest.raises(ValueError, match="lengths must match"):
        root_mean_squared_error([1], [1, 2])


def test_metric_inputs_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        nasa_asymmetric_score([1], [float("inf")])
