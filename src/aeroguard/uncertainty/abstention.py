"""Prediction abstention logic."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def apply_abstention(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    reasons: list[list[str]] = [[] for _ in range(len(result))]

    def add(mask: pd.Series | np.ndarray, reason: str) -> None:
        values = np.asarray(mask, dtype=bool)
        for idx, flag in enumerate(values):
            if flag:
                reasons[idx].append(reason)

    add(result.get("non_finite_input", pd.Series(False, index=result.index)), "NON_FINITE_INPUT")
    add(result.get("support_status", "") == "OUT_OF_SUPPORT", "OUT_OF_SUPPORT")
    add(result.get("feature_exceedance_fraction", 0.0).astype(float) > float(config["max_feature_exceedance_fraction"]), "FEATURE_RANGE_EXCEEDANCE")
    add(result.get("regime_distance", 0.0).astype(float) > float(config["max_regime_distance"]), "REGIME_DISTANCE_EXCESS")
    add(result.get("interval_width_ratio", 1.0).astype(float) > float(config["max_interval_width_ratio"]), "INTERVAL_WIDTH_EXCESS")
    add(result["predicted_rul"].astype(float) < float(config["min_plausible_rul"]), "POINT_RUL_BELOW_PLAUSIBLE_RANGE")
    add(result["predicted_rul"].astype(float) > float(config["max_plausible_rul"]), "POINT_RUL_ABOVE_PLAUSIBLE_RANGE")
    if "quantile_crossing_any" in result.columns and bool(config.get("abstain_on_quantile_crossing", True)):
        add(result["quantile_crossing_any"].astype(bool), "QUANTILE_CROSSING")
    result["abstain_flag"] = [len(item) > 0 for item in reasons]
    result["abstention_reason"] = [";".join(item) if item else "" for item in reasons]
    result["prediction_status"] = np.where(result["abstain_flag"], "abstained", "accepted")
    return result


def abstention_metrics(frame: pd.DataFrame, level: int = 90, high_error_threshold: float = 30.0) -> dict[str, float | int | None]:
    if frame.empty:
        return {}
    accepted = frame[~frame["abstain_flag"].astype(bool)]
    abstained = frame[frame["abstain_flag"].astype(bool)]

    def mae(data: pd.DataFrame) -> float | None:
        return None if data.empty else float(data["absolute_error"].mean())

    def rmse(data: pd.DataFrame) -> float | None:
        return None if data.empty else float(np.sqrt(np.mean(np.square(data["residual"]))))

    def cov(data: pd.DataFrame) -> float | None:
        return None if data.empty else float(data[f"covered_{level}"].astype(bool).mean())

    return {
        "engine_count": int(len(frame)),
        "accepted_count": int(len(accepted)),
        "abstained_count": int(len(abstained)),
        "acceptance_rate": float(len(accepted) / len(frame)),
        "abstention_rate": float(len(abstained) / len(frame)),
        "mae_before_abstention": mae(frame),
        "mae_after_abstention": mae(accepted),
        "rmse_before_abstention": rmse(frame),
        "rmse_after_abstention": rmse(accepted),
        "coverage_before_abstention": cov(frame),
        "coverage_after_abstention": cov(accepted),
        "mean_width_before_abstention": float(frame[f"interval_width_{level}"].mean()),
        "mean_width_after_abstention": None if accepted.empty else float(accepted[f"interval_width_{level}"].mean()),
        "error_rate_abstained": None if abstained.empty else float((abstained["absolute_error"] > high_error_threshold).mean()),
        "error_rate_accepted": None if accepted.empty else float((accepted["absolute_error"] > high_error_threshold).mean()),
        "high_error_predictions_abstained": int(((frame["absolute_error"] > high_error_threshold) & frame["abstain_flag"]).sum()),
        "low_error_predictions_unnecessarily_abstained": int(((frame["absolute_error"] <= high_error_threshold) & frame["abstain_flag"]).sum()),
    }
