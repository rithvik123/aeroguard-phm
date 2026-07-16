"""Temporal degradation constraints for physics-guided RUL training."""

from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F

Reduction = Literal["mean", "sum", "none"]
PenaltyKind = Literal["smooth_l1", "absolute", "squared", "l1"]


def _column(values: torch.Tensor, name: str) -> torch.Tensor:
    if values.ndim == 1:
        values = values.view(-1, 1)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError(f"{name} must have shape [n] or [n, 1].")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite.")
    return values


def _gap(values: torch.Tensor, expected: int | None = None) -> torch.Tensor:
    values = _column(values.to(dtype=torch.float32), "cycle_gap")
    if (values <= 0).any():
        raise ValueError("cycle_gap values must be positive.")
    if expected is not None and values.shape[0] != expected:
        raise ValueError("cycle_gap length does not match predictions.")
    return values


def _reduce(values: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "mean":
        return values.mean() if values.numel() else values.sum()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(f"Unsupported reduction: {reduction}")


def _penalty(values: torch.Tensor, kind: PenaltyKind) -> torch.Tensor:
    normalized = "absolute" if kind == "l1" else kind
    if normalized == "absolute":
        return values.abs()
    if normalized == "squared":
        return values.square()
    if normalized == "smooth_l1":
        return F.smooth_l1_loss(values, torch.zeros_like(values), reduction="none")
    raise ValueError(f"Unsupported penalty kind: {kind}")


def monotonicity_loss(
    earlier_prediction: torch.Tensor,
    later_prediction: torch.Tensor,
    *,
    tolerance: float = 0.0,
    cycle_gap: torch.Tensor | None = None,
    normalize_by_gap: bool = False,
    stage_weights: torch.Tensor | None = None,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Softly penalize later RUL predictions that exceed earlier predictions."""

    earlier = _column(earlier_prediction, "earlier_prediction")
    later = _column(later_prediction, "later_prediction")
    if earlier.shape != later.shape:
        raise ValueError("earlier_prediction and later_prediction must have the same shape.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    violation = torch.relu(later - earlier - float(tolerance))
    if normalize_by_gap:
        if cycle_gap is None:
            raise ValueError("cycle_gap is required when normalize_by_gap is true.")
        violation = violation / _gap(cycle_gap, earlier.shape[0]).to(device=violation.device, dtype=violation.dtype)
    if stage_weights is not None:
        weights = _column(stage_weights.to(device=violation.device, dtype=violation.dtype), "stage_weights")
        if weights.shape != violation.shape:
            raise ValueError("stage_weights must match prediction shape.")
        violation = violation * weights
    return _reduce(violation, reduction)


def monotonicity_diagnostics(
    earlier_prediction: torch.Tensor,
    later_prediction: torch.Tensor,
    *,
    tolerance: float = 0.0,
) -> dict[str, float]:
    earlier = _column(earlier_prediction.detach(), "earlier_prediction")
    later = _column(later_prediction.detach(), "later_prediction")
    violation = torch.relu(later - earlier - float(tolerance))
    count = int((violation > 0).sum().item())
    total = int(violation.numel())
    return {
        "monotonic_violation_count": float(count),
        "monotonic_violation_rate": float(count / total) if total else 0.0,
        "monotonic_mean_violation": float(violation.mean().item()) if total else 0.0,
        "monotonic_max_violation": float(violation.max().item()) if total else 0.0,
    }


def cycle_rate_consistency_loss(
    earlier_prediction: torch.Tensor,
    later_prediction: torch.Tensor,
    cycle_gap: torch.Tensor,
    *,
    kind: PenaltyKind = "smooth_l1",
    tolerance: float = 0.0,
    clip: float | None = None,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Penalize deviation from approximately one cycle of RUL loss per cycle."""

    earlier = _column(earlier_prediction, "earlier_prediction")
    later = _column(later_prediction, "later_prediction")
    if earlier.shape != later.shape:
        raise ValueError("earlier_prediction and later_prediction must have the same shape.")
    gap = _gap(cycle_gap, earlier.shape[0]).to(device=earlier.device, dtype=earlier.dtype)
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    residual = later - earlier + gap
    residual = residual.sign() * torch.relu(residual.abs() - float(tolerance))
    if clip is not None:
        if clip <= 0:
            raise ValueError("clip must be positive when provided.")
        residual = residual.clamp(min=-float(clip), max=float(clip))
    return _reduce(_penalty(residual, kind), reduction)


def cycle_rate_diagnostics(
    earlier_prediction: torch.Tensor,
    later_prediction: torch.Tensor,
    cycle_gap: torch.Tensor,
    *,
    tolerance: float = 0.0,
) -> dict[str, float]:
    earlier = _column(earlier_prediction.detach(), "earlier_prediction")
    later = _column(later_prediction.detach(), "later_prediction")
    gap = _gap(cycle_gap.detach(), earlier.shape[0]).to(device=earlier.device, dtype=earlier.dtype)
    residual = (later - earlier + gap).view(-1)
    abs_residual = residual.abs()
    total = int(abs_residual.numel())
    violation = abs_residual > float(tolerance)
    return {
        "rate_mean_residual": float(residual.mean().item()) if total else 0.0,
        "rate_median_residual": float(residual.median().item()) if total else 0.0,
        "rate_residual_std": float(residual.std(unbiased=False).item()) if total else 0.0,
        "rate_violation_count": float(violation.sum().item()),
        "rate_violation_rate": float(violation.to(dtype=torch.float32).mean().item()) if total else 0.0,
    }


def smoothness_loss(
    earlier_prediction: torch.Tensor,
    middle_prediction: torch.Tensor,
    later_prediction: torch.Tensor,
    *,
    left_gap: torch.Tensor | None = None,
    right_gap: torch.Tensor | None = None,
    kind: PenaltyKind = "smooth_l1",
    tolerance: float = 0.0,
    gap_normalized: bool = False,
    late_life_weights: torch.Tensor | None = None,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Penalize large second differences across same-engine triplets."""

    earlier = _column(earlier_prediction, "earlier_prediction")
    middle = _column(middle_prediction, "middle_prediction")
    later = _column(later_prediction, "later_prediction")
    if earlier.shape != middle.shape or earlier.shape != later.shape:
        raise ValueError("Triplet predictions must have matching shapes.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    if left_gap is None and right_gap is None:
        second = later - (2.0 * middle) + earlier
    else:
        if left_gap is None or right_gap is None:
            raise ValueError("Both left_gap and right_gap are required.")
        left = _gap(left_gap, earlier.shape[0]).to(device=earlier.device, dtype=earlier.dtype)
        right = _gap(right_gap, earlier.shape[0]).to(device=earlier.device, dtype=earlier.dtype)
        if not gap_normalized and not torch.equal(left, right):
            raise ValueError("Unequal-gap triplets require gap_normalized=True.")
        if gap_normalized:
            second = ((later - middle) / right) - ((middle - earlier) / left)
        else:
            second = later - (2.0 * middle) + earlier
    residual = second.sign() * torch.relu(second.abs() - float(tolerance))
    values = _penalty(residual, kind)
    if late_life_weights is not None:
        weights = _column(late_life_weights.to(device=values.device, dtype=values.dtype), "late_life_weights")
        if weights.shape != values.shape:
            raise ValueError("late_life_weights must match prediction shape.")
        values = values * weights
    return _reduce(values, reduction)


def smoothness_diagnostics(
    earlier_prediction: torch.Tensor,
    middle_prediction: torch.Tensor,
    later_prediction: torch.Tensor,
    *,
    tolerance: float = 0.0,
) -> dict[str, float]:
    earlier = _column(earlier_prediction.detach(), "earlier_prediction")
    middle = _column(middle_prediction.detach(), "middle_prediction")
    later = _column(later_prediction.detach(), "later_prediction")
    second = (later - (2.0 * middle) + earlier).abs().view(-1)
    total = int(second.numel())
    violation = second > float(tolerance)
    return {
        "smooth_mean_abs_second_difference": float(second.mean().item()) if total else 0.0,
        "smooth_max_second_difference": float(second.max().item()) if total else 0.0,
        "smooth_violation_rate": float(violation.to(dtype=torch.float32).mean().item()) if total else 0.0,
    }
