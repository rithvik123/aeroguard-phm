"""Frozen critical-boundary safety guard."""

from __future__ import annotations


def apply_critical_boundary_guard(
    base_rul: float,
    *,
    boundary_low: float = 15.0,
    boundary_high: float = 25.0,
    margin: float = 0.5,
    bound: float = 10.0,
) -> dict[str, float | bool]:
    """Apply the locked one-sided critical-boundary rule.

    The guard is deterministic and downward-only:
    active when ``boundary_low < base_rul <= boundary_high``.
    """

    base = max(0.0, float(base_rul))
    active = bool(boundary_low < base <= boundary_high)
    correction = min(float(bound), max(0.0, base - float(boundary_low) + float(margin))) if active else 0.0
    adjusted = max(0.0, base - correction)
    return {
        "base_rul": base,
        "safety_adjusted_rul": adjusted,
        "correction_cycles": correction,
        "safety_guard_activated": active,
    }
