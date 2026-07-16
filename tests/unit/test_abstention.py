import pandas as pd

from aeroguard.uncertainty.abstention import abstention_metrics, apply_abstention


def _config() -> dict:
    return {
        "max_feature_exceedance_fraction": 0.2,
        "max_regime_distance": 5.0,
        "max_interval_width_ratio": 2.0,
        "min_plausible_rul": 0.0,
        "max_plausible_rul": 200.0,
        "abstain_on_quantile_crossing": True,
    }


def test_apply_abstention_no_reason_and_multiple_reasons() -> None:
    frame = pd.DataFrame(
        {
            "predicted_rul": [50.0, 300.0],
            "support_status": ["IN_SUPPORT", "OUT_OF_SUPPORT"],
            "feature_exceedance_fraction": [0.0, 0.5],
            "regime_distance": [1.0, 8.0],
            "interval_width_ratio": [1.0, 3.0],
            "non_finite_input": [False, False],
            "quantile_crossing_any": [False, True],
        }
    )

    result = apply_abstention(frame, _config())

    assert result["abstain_flag"].tolist() == [False, True]
    assert "OUT_OF_SUPPORT" in result["abstention_reason"].iloc[1]
    assert "INTERVAL_WIDTH_EXCESS" in result["abstention_reason"].iloc[1]


def test_abstention_metrics() -> None:
    frame = pd.DataFrame(
        {
            "abstain_flag": [False, True],
            "absolute_error": [5.0, 40.0],
            "residual": [5.0, 40.0],
            "covered_90": [True, False],
            "interval_width_90": [10.0, 50.0],
        }
    )

    metrics = abstention_metrics(frame, high_error_threshold=30.0)

    assert metrics["abstention_rate"] == 0.5
    assert metrics["high_error_predictions_abstained"] == 1
