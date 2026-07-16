"""Reusable scalar loss components for Phase 5C physics-guided training."""

from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F

LossKind = Literal["smooth_l1", "mse", "mae", "absolute", "squared"]


def _column(values: torch.Tensor, name: str) -> torch.Tensor:
    if values.ndim == 1:
        values = values.view(-1, 1)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError(f"{name} must have shape [n] or [n, 1].")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite.")
    return values


def _loss_from_residual(residual: torch.Tensor, kind: LossKind) -> torch.Tensor:
    normalized = "mae" if kind == "absolute" else "mse" if kind == "squared" else kind
    if normalized == "smooth_l1":
        return F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none")
    if normalized == "mse":
        return residual.square()
    if normalized == "mae":
        return residual.abs()
    raise ValueError(f"Unsupported loss kind: {kind}")


def primary_rul_loss(prediction: torch.Tensor, target: torch.Tensor, *, kind: LossKind = "smooth_l1") -> torch.Tensor:
    """Primary Phase 5B-compatible RUL regression loss."""

    pred = _column(prediction, "prediction")
    true = _column(target.to(device=pred.device, dtype=pred.dtype), "target")
    if pred.shape != true.shape:
        raise ValueError("prediction and target must have matching shapes.")
    return _loss_from_residual(pred - true, kind).mean()


def nonnegative_rul_penalty(rul_raw: torch.Tensor) -> torch.Tensor:
    """Penalty that keeps raw RUL diagnostics visible while discouraging negatives."""

    raw = _column(rul_raw, "rul_raw")
    return torch.relu(-raw).mean()


def health_proxy_loss(predicted_health: torch.Tensor, target_health: torch.Tensor, *, kind: LossKind = "mse") -> torch.Tensor:
    pred = _column(predicted_health, "predicted_health")
    true = _column(target_health.to(device=pred.device, dtype=pred.dtype), "target_health")
    if pred.shape != true.shape:
        raise ValueError("predicted_health and target_health must have matching shapes.")
    if ((true < 0.0) | (true > 1.0)).any():
        raise ValueError("target_health must be in [0, 1].")
    return _loss_from_residual(pred - true, kind).mean()


def health_monotonicity_loss(
    earlier_health: torch.Tensor,
    later_health: torch.Tensor,
    *,
    tolerance: float = 0.0,
) -> torch.Tensor:
    """Softly require an earlier same-engine health proxy to be at least later health."""

    earlier = _column(earlier_health, "earlier_health")
    later = _column(later_health, "later_health")
    if earlier.shape != later.shape:
        raise ValueError("earlier_health and later_health must have matching shapes.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    return torch.relu(later - earlier - float(tolerance)).mean()


def optimistic_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    kind: LossKind = "mae",
    severe_threshold: float | None = None,
    severe_multiplier: float = 1.0,
) -> torch.Tensor:
    """Safety-oriented asymmetric loss for over-predicted RUL."""

    pred = _column(prediction, "prediction")
    true = _column(target.to(device=pred.device, dtype=pred.dtype), "target")
    if pred.shape != true.shape:
        raise ValueError("prediction and target must have matching shapes.")
    if severe_threshold is not None and severe_threshold < 0:
        raise ValueError("severe_threshold must be non-negative.")
    if severe_multiplier < 1.0:
        raise ValueError("severe_multiplier must be at least 1.")
    optimistic = torch.relu(pred - true)
    values = _loss_from_residual(optimistic, kind)
    if severe_threshold is not None:
        multiplier = torch.where(
            optimistic > float(severe_threshold),
            torch.full_like(values, float(severe_multiplier)),
            torch.ones_like(values),
        )
        values = values * multiplier
    return values.mean()


def degradation_rate_head_loss(
    predicted_rate: torch.Tensor,
    earlier_target: torch.Tensor,
    later_target: torch.Tensor,
    cycle_gap: torch.Tensor,
    *,
    kind: LossKind = "smooth_l1",
    mask_plateau: bool = True,
    plateau_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Train the optional rate head from target RUL decrease per cycle."""

    pred = _column(predicted_rate, "predicted_rate")
    earlier = _column(earlier_target.to(device=pred.device, dtype=pred.dtype), "earlier_target")
    later = _column(later_target.to(device=pred.device, dtype=pred.dtype), "later_target")
    gap = _column(cycle_gap.to(device=pred.device, dtype=pred.dtype), "cycle_gap")
    if pred.shape != earlier.shape or pred.shape != later.shape or pred.shape != gap.shape:
        raise ValueError("rate-head inputs must have matching shapes.")
    if (gap <= 0).any():
        raise ValueError("cycle_gap values must be positive.")
    target_rate = torch.relu((earlier - later) / gap)
    values = _loss_from_residual(pred - target_rate, kind)
    if mask_plateau and plateau_mask is not None:
        mask = _column(plateau_mask.to(device=pred.device, dtype=torch.float32), "plateau_mask")
        if mask.shape != values.shape:
            raise ValueError("plateau_mask must match rate-head inputs.")
        keep = (mask <= 0.0).to(dtype=values.dtype)
        if keep.sum() <= 0:
            return values.sum() * 0.0
        values = values * keep
        return values.sum() / keep.sum().clamp_min(1.0)
    return values.mean()
