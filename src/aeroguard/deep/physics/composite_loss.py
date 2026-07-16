"""Composite configurable loss for the Phase 5C physics-guided model."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any

import torch
from torch import nn

from aeroguard.deep.physics.loss_components import (
    degradation_rate_head_loss,
    health_monotonicity_loss,
    health_proxy_loss,
    nonnegative_rul_penalty,
    optimistic_prediction_loss,
    primary_rul_loss,
)
from aeroguard.deep.physics.regime_consistency import latent_consistency_loss, prediction_consistency_loss
from aeroguard.deep.physics.temporal_constraints import (
    cycle_rate_consistency_loss,
    cycle_rate_diagnostics,
    monotonicity_diagnostics,
    monotonicity_loss,
    smoothness_diagnostics,
    smoothness_loss,
)


@dataclass(frozen=True)
class PhysicsLossConfig:
    lambda_data: float = 1.0
    lambda_monotonic: float = 0.0
    lambda_rate: float = 0.0
    lambda_smooth: float = 0.0
    lambda_health: float = 0.0
    lambda_health_monotonic: float = 0.0
    lambda_regime: float = 0.0
    lambda_nonnegative: float = 0.0
    lambda_optimistic: float = 0.0
    data_loss: str = "smooth_l1"
    rate_loss: str = "smooth_l1"
    smooth_loss: str = "smooth_l1"
    health_loss: str = "mse"
    optimistic_loss: str = "mae"
    monotonic_tolerance: float = 0.0
    rate_tolerance: float = 0.0
    smoothness_tolerance: float = 0.0
    health_monotonic_tolerance: float = 0.0
    regime_tolerance: float = 0.0
    regime_metric: str = "cosine"
    regime_formulation: str = "latent"
    severe_optimistic_threshold: float = 30.0
    severe_optimistic_multiplier: float = 1.0
    normalize_monotonic_by_gap: bool = False
    include_rate_head_loss: bool = False
    mask_rate_plateau: bool = True
    allow_missing_optional_batches: bool = False

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "PhysicsLossConfig":
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})


class CompositePhysicsLoss(nn.Module):
    """Compute data, physics, auxiliary, and safety losses from model outputs."""

    def __init__(self, config: PhysicsLossConfig | dict[str, Any]) -> None:
        super().__init__()
        self.config = config if isinstance(config, PhysicsLossConfig) else PhysicsLossConfig.from_mapping(config)
        self._validate_config()

    def forward(self, outputs: dict[str, torch.Tensor | None], batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        prediction = _required_output(outputs, "rul_prediction")
        target = _required_batch(batch, "target_rul", prediction).to(device=prediction.device, dtype=prediction.dtype)
        zero = prediction.sum() * 0.0
        raw: dict[str, torch.Tensor] = {
            "data_loss": primary_rul_loss(prediction, target, kind=self.config.data_loss),
            "monotonic_loss": zero,
            "rate_loss": zero,
            "smooth_loss": zero,
            "health_loss": zero,
            "health_monotonic_loss": zero,
            "regime_loss": zero,
            "nonnegative_loss": zero,
            "safety_optimistic_loss": zero,
        }
        diagnostics: dict[str, float | str | list[str]] = {"skipped_terms": []}

        pair = _optional_indices(batch, "pair_indices", prediction.device)
        pair_gap = _optional_column(batch, "pair_cycle_gaps", prediction)
        if self.config.lambda_monotonic > 0.0:
            if _missing_pair(pair, pair_gap):
                self._missing("monotonic_loss", diagnostics)
            else:
                earlier, later = prediction[pair[:, 0]], prediction[pair[:, 1]]
                raw["monotonic_loss"] = monotonicity_loss(
                    earlier,
                    later,
                    tolerance=self.config.monotonic_tolerance,
                    cycle_gap=pair_gap,
                    normalize_by_gap=self.config.normalize_monotonic_by_gap,
                )
                diagnostics.update(monotonicity_diagnostics(earlier, later, tolerance=self.config.monotonic_tolerance))

        if self.config.lambda_rate > 0.0:
            if _missing_pair(pair, pair_gap):
                self._missing("rate_loss", diagnostics)
            else:
                earlier, later = prediction[pair[:, 0]], prediction[pair[:, 1]]
                rate_loss = cycle_rate_consistency_loss(
                    earlier,
                    later,
                    pair_gap,
                    kind=self.config.rate_loss,
                    tolerance=self.config.rate_tolerance,
                )
                if self.config.include_rate_head_loss:
                    rate = _required_output(outputs, "degradation_rate")
                    earlier_target = target[pair[:, 0]]
                    later_target = target[pair[:, 1]]
                    plateau_mask = batch.get("pair_plateau_mask")
                    plateau = None if plateau_mask is None else _as_column(plateau_mask, prediction).to(device=prediction.device)
                    rate_loss = rate_loss + degradation_rate_head_loss(
                        rate[pair[:, 0]],
                        earlier_target,
                        later_target,
                        pair_gap,
                        kind=self.config.rate_loss,
                        mask_plateau=self.config.mask_rate_plateau,
                        plateau_mask=plateau,
                    )
                raw["rate_loss"] = rate_loss
                diagnostics.update(cycle_rate_diagnostics(earlier, later, pair_gap, tolerance=self.config.rate_tolerance))

        triplet = _optional_indices(batch, "triplet_indices", prediction.device)
        left_gap = _optional_column(batch, "triplet_left_gaps", prediction)
        right_gap = _optional_column(batch, "triplet_right_gaps", prediction)
        if self.config.lambda_smooth > 0.0:
            if triplet is None or triplet.numel() == 0:
                self._missing("smooth_loss", diagnostics)
            else:
                raw["smooth_loss"] = smoothness_loss(
                    prediction[triplet[:, 0]],
                    prediction[triplet[:, 1]],
                    prediction[triplet[:, 2]],
                    left_gap=left_gap,
                    right_gap=right_gap,
                    kind=self.config.smooth_loss,
                    tolerance=self.config.smoothness_tolerance,
                )
                diagnostics.update(
                    smoothness_diagnostics(
                        prediction[triplet[:, 0]],
                        prediction[triplet[:, 1]],
                        prediction[triplet[:, 2]],
                        tolerance=self.config.smoothness_tolerance,
                    )
                )

        if self.config.lambda_health > 0.0:
            health = _required_output(outputs, "health_score")
            health_target = _required_batch(batch, "health_target", prediction).to(device=prediction.device, dtype=prediction.dtype)
            raw["health_loss"] = health_proxy_loss(health, health_target, kind=self.config.health_loss)

        if self.config.lambda_health_monotonic > 0.0:
            health = _required_output(outputs, "health_score")
            if _missing_pair(pair, pair_gap):
                self._missing("health_monotonic_loss", diagnostics)
            else:
                raw["health_monotonic_loss"] = health_monotonicity_loss(
                    health[pair[:, 0]],
                    health[pair[:, 1]],
                    tolerance=self.config.health_monotonic_tolerance,
                )

        if self.config.lambda_regime > 0.0:
            regime = _optional_indices(batch, "regime_pair_indices", prediction.device)
            if regime is None or regime.numel() == 0:
                self._missing("regime_loss", diagnostics)
            else:
                pieces = []
                if self.config.regime_formulation in {"latent", "both"}:
                    latent = _required_output(outputs, "latent")
                    pieces.append(
                        latent_consistency_loss(
                            latent[regime[:, 0]],
                            latent[regime[:, 1]],
                            metric=self.config.regime_metric,
                            tolerance=self.config.regime_tolerance,
                        )
                    )
                if self.config.regime_formulation in {"prediction", "both"}:
                    pieces.append(
                        prediction_consistency_loss(
                            prediction[regime[:, 0]],
                            prediction[regime[:, 1]],
                            tolerance=self.config.regime_tolerance,
                        )
                    )
                if not pieces:
                    raise ValueError(f"Unsupported regime_formulation: {self.config.regime_formulation}")
                raw["regime_loss"] = sum(pieces) / float(len(pieces))

        if self.config.lambda_nonnegative > 0.0:
            raw["nonnegative_loss"] = nonnegative_rul_penalty(_required_output(outputs, "rul_raw"))

        if self.config.lambda_optimistic > 0.0:
            raw["safety_optimistic_loss"] = optimistic_prediction_loss(
                prediction,
                target,
                kind=self.config.optimistic_loss,
                severe_threshold=self.config.severe_optimistic_threshold,
                severe_multiplier=self.config.severe_optimistic_multiplier,
            )

        weighted = {
            "data_loss": raw["data_loss"] * self.config.lambda_data,
            "monotonic_loss": raw["monotonic_loss"] * self.config.lambda_monotonic,
            "rate_loss": raw["rate_loss"] * self.config.lambda_rate,
            "smooth_loss": raw["smooth_loss"] * self.config.lambda_smooth,
            "health_loss": raw["health_loss"] * self.config.lambda_health,
            "health_monotonic_loss": raw["health_monotonic_loss"] * self.config.lambda_health_monotonic,
            "regime_loss": raw["regime_loss"] * self.config.lambda_regime,
            "nonnegative_loss": raw["nonnegative_loss"] * self.config.lambda_nonnegative,
            "safety_optimistic_loss": raw["safety_optimistic_loss"] * self.config.lambda_optimistic,
        }
        total = sum(weighted.values(), zero)
        if not torch.isfinite(total):
            raise RuntimeError("Composite physics loss is non-finite.")
        return {"total_loss": total, "raw_terms": raw, "weighted_terms": weighted, "diagnostics": diagnostics}

    def _missing(self, term: str, diagnostics: dict[str, Any]) -> None:
        if self.config.allow_missing_optional_batches:
            diagnostics.setdefault("skipped_terms", []).append(term)
            return
        raise ValueError(f"{term} is active but its required structured batch is missing.")

    def _validate_config(self) -> None:
        weights = {
            "lambda_data": self.config.lambda_data,
            "lambda_monotonic": self.config.lambda_monotonic,
            "lambda_rate": self.config.lambda_rate,
            "lambda_smooth": self.config.lambda_smooth,
            "lambda_health": self.config.lambda_health,
            "lambda_health_monotonic": self.config.lambda_health_monotonic,
            "lambda_regime": self.config.lambda_regime,
            "lambda_nonnegative": self.config.lambda_nonnegative,
            "lambda_optimistic": self.config.lambda_optimistic,
        }
        for name, value in weights.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if float(self.config.lambda_data) <= 0.0:
            raise ValueError("lambda_data must be greater than zero.")
        for name in [
            "monotonic_tolerance",
            "rate_tolerance",
            "smoothness_tolerance",
            "health_monotonic_tolerance",
            "regime_tolerance",
            "severe_optimistic_threshold",
        ]:
            if float(getattr(self.config, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if float(self.config.severe_optimistic_multiplier) < 1.0:
            raise ValueError("severe_optimistic_multiplier must be at least 1.")


def _required_output(outputs: dict[str, torch.Tensor | None], key: str) -> torch.Tensor:
    value = outputs.get(key)
    if value is None:
        raise ValueError(f"Model output '{key}' is required for the active loss.")
    if not torch.is_tensor(value) or not torch.isfinite(value).all():
        raise ValueError(f"Model output '{key}' must be a finite tensor.")
    return value


def _required_batch(batch: dict[str, torch.Tensor], key: str, reference: torch.Tensor) -> torch.Tensor:
    if key not in batch:
        raise ValueError(f"Batch field '{key}' is required.")
    return _as_column(batch[key], reference)


def _as_column(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.to(device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 1:
        tensor = tensor.view(-1, 1)
    if tensor.ndim != 2 or tensor.shape[1] != 1:
        raise ValueError("Batch scalar fields must have shape [n] or [n, 1].")
    if not torch.isfinite(tensor).all():
        raise ValueError("Batch scalar fields must be finite.")
    return tensor


def _optional_column(batch: dict[str, torch.Tensor], key: str, reference: torch.Tensor) -> torch.Tensor | None:
    if key not in batch or batch[key] is None:
        return None
    return _as_column(batch[key], reference)


def _optional_indices(batch: dict[str, torch.Tensor], key: str, device: torch.device) -> torch.Tensor | None:
    if key not in batch or batch[key] is None:
        return None
    value = batch[key] if torch.is_tensor(batch[key]) else torch.as_tensor(batch[key])
    value = value.to(device=device, dtype=torch.long)
    if value.ndim != 2 or value.shape[1] not in {2, 3}:
        raise ValueError(f"{key} must have shape [n, 2] or [n, 3].")
    return value


def _missing_pair(pair: torch.Tensor | None, gap: torch.Tensor | None) -> bool:
    return pair is None or pair.numel() == 0 or gap is None or gap.numel() == 0
