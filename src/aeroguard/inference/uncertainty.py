"""Frozen conformal uncertainty helpers."""

from __future__ import annotations


def conformal_intervals(predicted_rul: float, uncertainty_config: dict[str, object]) -> dict[str, float]:
    radii = uncertainty_config.get("radii", {}) if isinstance(uncertainty_config, dict) else {}
    result: dict[str, float] = {}
    for key, value in sorted(radii.items(), key=lambda item: float(item[0])):
        level = int(round(float(key) * 100))
        radius = float(value)
        lower = max(0.0, float(predicted_rul) - radius)
        upper = float(predicted_rul) + radius
        result[f"lower_{level}"] = lower
        result[f"upper_{level}"] = upper
        result[f"interval_width_{level}"] = upper - lower
    return result
