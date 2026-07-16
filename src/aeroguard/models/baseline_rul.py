"""Classical scikit-learn RUL baseline models."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def build_baseline_models(config: Mapping[str, object]) -> dict[str, object]:
    """Build the configured classical baseline models."""
    ridge_params = dict(config.get("ridge", {}))
    forest_params = dict(config.get("random_forest", {}))
    models = {
        "dummy_median": DummyRegressor(strategy="median"),
        "ridge": Ridge(**ridge_params),
        "random_forest": RandomForestRegressor(**forest_params),
    }
    return models


def fit_models(
    models: Mapping[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, object]:
    """Fit all provided models and return them by name."""
    fitted: dict[str, object] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        fitted[name] = model
    return fitted


def predict_models(
    models: Mapping[str, object],
    x_values: np.ndarray,
    clip_non_negative: bool = True,
) -> dict[str, np.ndarray]:
    """Generate model predictions with optional physical non-negative clipping."""
    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        y_pred = np.asarray(model.predict(x_values), dtype=float)
        if clip_non_negative:
            y_pred = np.maximum(y_pred, 0.0)
        predictions[name] = y_pred
    return predictions
