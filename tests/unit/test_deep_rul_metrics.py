import numpy as np
import pandas as pd
import pytest

from aeroguard.evaluation.deep_rul_metrics import deep_point_metrics, metrics_by_group, prediction_direction


def test_deep_point_metrics_include_directional_error_rates() -> None:
    metrics = deep_point_metrics([10.0, 20.0, 30.0], [12.0, 18.0, 65.0], severe_optimistic_threshold=30.0)

    assert metrics["mae"] == pytest.approx(13.0)
    assert metrics["mean_signed_error"] == pytest.approx((2.0 - 2.0 + 35.0) / 3.0)
    assert metrics["optimistic_prediction_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["conservative_prediction_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["severe_optimistic_error_rate"] == pytest.approx(1.0 / 3.0)
    assert np.isfinite(metrics["nasa_score"])


def test_metrics_by_group_and_prediction_direction() -> None:
    frame = pd.DataFrame(
        {
            "subset": ["FD001", "FD001", "FD002"],
            "true_rul": [5.0, 10.0, 15.0],
            "predicted_rul": [6.0, 8.0, 15.0],
        }
    )

    grouped = metrics_by_group(frame, "subset")

    assert set(grouped["subset"]) == {"FD001", "FD002"}
    assert prediction_direction(1.0) == "over_prediction_optimistic"
    assert prediction_direction(-1.0) == "under_prediction_conservative"
    assert prediction_direction(0.0) == "perfect"
    with pytest.raises(ValueError):
        deep_point_metrics([], [])

