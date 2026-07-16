"""Quantile Gradient Boosting intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from aeroguard.uncertainty.tree_quantiles import interval_quantiles_for_level


def quantile_gradient_boosting_available() -> tuple[bool, str]:
    try:
        GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=2).fit([[0.0], [1.0]], [0.0, 1.0])
    except Exception as exc:  # pragma: no cover - version dependent guard
        return False, f"{type(exc).__name__}: {exc}"
    return True, "GradientBoostingRegressor(loss='quantile') is available."


@dataclass
class QuantileGradientBoostingIntervals:
    nominal_levels: list[float]
    parameters: dict[str, Any]
    random_state: int = 42

    def _model(self, alpha: float) -> GradientBoostingRegressor:
        params = dict(self.parameters)
        params["loss"] = "quantile"
        params["alpha"] = float(alpha)
        params["random_state"] = int(self.random_state)
        return GradientBoostingRegressor(**params)

    def fit(self, x_values: object, y_values: Iterable[float]) -> "QuantileGradientBoostingIntervals":
        ok, reason = quantile_gradient_boosting_available()
        if not ok:
            raise RuntimeError(reason)
        y = np.asarray(list(y_values), dtype=float)
        if len(y) == 0 or not np.isfinite(y).all():
            raise ValueError("Quantile regression targets must be non-empty and finite.")
        alphas = sorted({q for level in self.nominal_levels for q in interval_quantiles_for_level(level)})
        self.models_: dict[float, GradientBoostingRegressor] = {}
        for alpha in alphas:
            self.models_[float(alpha)] = self._model(float(alpha)).fit(x_values, y)
        return self

    def predict_interval_frame(self, x_values: object, prefix: str = "") -> pd.DataFrame:
        if not hasattr(self, "models_"):
            raise RuntimeError("QuantileGradientBoostingIntervals must be fitted before predict.")
        predictions = {alpha: model.predict(x_values).astype(float) for alpha, model in self.models_.items()}
        n_rows = len(next(iter(predictions.values()))) if predictions else 0
        output: dict[str, np.ndarray] = {}
        for level in self.nominal_levels:
            low_q, high_q = interval_quantiles_for_level(float(level))
            pct = int(round(float(level) * 100))
            lower_raw = predictions[float(low_q)]
            upper_raw = predictions[float(high_q)]
            crossing = lower_raw > upper_raw
            ordered = np.sort(np.column_stack([lower_raw, upper_raw]), axis=1)
            output[f"{prefix}lower_{pct}"] = np.maximum(0.0, ordered[:, 0])
            output[f"{prefix}upper_{pct}"] = ordered[:, 1]
            output[f"{prefix}raw_lower_{pct}"] = lower_raw
            output[f"{prefix}raw_upper_{pct}"] = upper_raw
            output[f"{prefix}quantile_crossing_{pct}"] = crossing.astype(bool)
        output[f"{prefix}quantile_crossing_any"] = (
            np.column_stack([output[key] for key in output if key.startswith(f"{prefix}quantile_crossing_")]).any(axis=1)
            if n_rows
            else np.array([], dtype=bool)
        )
        return pd.DataFrame(output)

    def metadata(self) -> dict[str, object]:
        return {
            "nominal_levels": self.nominal_levels,
            "parameters": self.parameters,
            "trained_alphas": sorted(self.models_),
            "availability": quantile_gradient_boosting_available()[1],
        }
