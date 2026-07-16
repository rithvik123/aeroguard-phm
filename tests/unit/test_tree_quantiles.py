import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from aeroguard.uncertainty.tree_quantiles import interval_quantiles_for_level, tree_prediction_matrix, tree_quantile_interval_frame


def _model() -> RandomForestRegressor:
    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.arange(20, dtype=float)
    return RandomForestRegressor(n_estimators=5, random_state=3, max_depth=3).fit(x, y)


def test_tree_quantile_intervals_are_ordered_and_deterministic() -> None:
    model = _model()
    frame1 = tree_quantile_interval_frame(model, [[1.0], [5.0]], [0.8, 0.9, 0.95], prefix="tree_")
    frame2 = tree_quantile_interval_frame(model, [[1.0], [5.0]], [0.8, 0.9, 0.95], prefix="tree_")

    assert frame1.equals(frame2)
    assert frame1["tree_tree_count"].tolist() == [5, 5]
    assert (frame1["tree_lower_90"] <= frame1["tree_upper_90"]).all()


def test_tree_prediction_matrix_and_invalid_level() -> None:
    matrix = tree_prediction_matrix(_model(), [[1.0], [2.0]])
    assert matrix.shape == (2, 5)

    with pytest.raises(ValueError):
        interval_quantiles_for_level(1.2)
