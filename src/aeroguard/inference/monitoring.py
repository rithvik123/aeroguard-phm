"""Inference monitoring records and lightweight drift summaries."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any


INFERENCE_LOG_FIELDS = [
    "timestamp",
    "engine_id",
    "model_version",
    "input_schema_hash",
    "input_row_count",
    "operating_regime",
    "base_rul",
    "adjusted_rul",
    "correction_amount",
    "interval_width",
    "support_score",
    "guard_activation",
    "review_status",
    "maintenance_action",
    "latency_ms",
    "warning_count",
]


MONITORING_COMPONENTS = [
    "input_schema_drift",
    "missing_sensor_rate",
    "feature_range_violations",
    "regime_distribution_drift",
    "support_score_drift",
    "prediction_distribution_drift",
    "interval_width_drift",
    "safety_guard_activation_rate",
    "review_rate",
    "maintenance_action_distribution",
    "rul_trajectory_instability",
    "inference_latency",
    "runtime_failures",
]


def schema_hash(columns: list[str]) -> str:
    text = "\n".join(columns)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_inference_log(
    prediction: dict[str, Any],
    *,
    input_columns: list[str],
    input_row_count: int,
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine_id": prediction.get("engine_id"),
        "model_version": prediction.get("model_version"),
        "input_schema_hash": schema_hash(input_columns),
        "input_row_count": int(input_row_count),
        "operating_regime": prediction.get("operating_regime"),
        "base_rul": prediction.get("base_rul"),
        "adjusted_rul": prediction.get("safety_adjusted_rul"),
        "correction_amount": prediction.get("correction_cycles"),
        "interval_width": prediction.get("interval_width_90"),
        "support_score": prediction.get("support_score"),
        "guard_activation": prediction.get("safety_guard_activated"),
        "review_status": prediction.get("review_required"),
        "maintenance_action": prediction.get("maintenance_action"),
        "latency_ms": float(latency_ms),
        "warning_count": len(prediction.get("warnings", []) or []),
    }


def monitoring_spec() -> dict[str, Any]:
    return {
        "components": MONITORING_COMPONENTS,
        "inference_log_fields": INFERENCE_LOG_FIELDS,
        "raw_sensitive_data_logged_by_default": False,
    }
