import numpy as np

from aeroguard.uncertainty.quantile_regression import QuantileGradientBoostingIntervals, quantile_gradient_boosting_available


def test_quantile_gradient_boosting_availability() -> None:
    available, reason = quantile_gradient_boosting_available()

    assert isinstance(available, bool)
    assert reason


def test_quantile_gradient_boosting_models_and_crossing_correction() -> None:
    x = np.arange(30, dtype=float).reshape(-1, 1)
    y = x.ravel() * 0.5
    model = QuantileGradientBoostingIntervals(
        nominal_levels=[0.8, 0.9],
        parameters={"n_estimators": 4, "learning_rate": 0.1, "max_depth": 1, "min_samples_leaf": 1},
        random_state=4,
    ).fit(x, y)

    intervals = model.predict_interval_frame([[5.0], [10.0]], prefix="qgb_")

    assert {"qgb_lower_80", "qgb_upper_80", "qgb_quantile_crossing_any"}.issubset(intervals.columns)
    assert (intervals["qgb_lower_90"] <= intervals["qgb_upper_90"]).all()
    assert np.isfinite(intervals[["qgb_lower_80", "qgb_upper_80"]].to_numpy()).all()
