"""Bounded Phase 5C candidate registry definitions and validation."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


DEFAULT_LOSS_WEIGHTS = {
    "lambda_data": 1.0,
    "lambda_monotonic": 0.0,
    "lambda_rate": 0.0,
    "lambda_smooth": 0.0,
    "lambda_health": 0.0,
    "lambda_health_monotonic": 0.0,
    "lambda_regime": 0.0,
    "lambda_nonnegative": 0.0,
    "lambda_optimistic": 0.0,
}


def default_candidate_registry() -> list[dict[str, Any]]:
    """Return the fixed default Phase 5C ablation registry."""

    base_architecture = {
        "architecture": "physics_guided_patch_transformer",
        "window_length": 50,
        "patch_length": 10,
        "patch_stride": 5,
        "projection_dim": 64,
        "layers": 2,
        "heads": 4,
        "feedforward_dim": 192,
        "dropout": 0.15,
        "positional_encoding": "learnable",
        "pooling": "mean",
        "causal_attention": False,
    }
    candidates = [
        _candidate("phase5b_reimplementation_baseline", base_architecture, heads=[], losses=["data"], weights={}),
        _candidate("physics_monotonic", base_architecture, heads=[], losses=["data", "monotonic"], weights={"lambda_monotonic": 0.05}, pairs=["adjacent"]),
        _candidate("physics_cycle_rate", base_architecture, heads=["rate"], losses=["data", "rate"], weights={"lambda_rate": 0.03}, pairs=["fixed_gap"]),
        _candidate("physics_smooth", base_architecture, heads=[], losses=["data", "smooth"], weights={"lambda_smooth": 0.02}, pairs=["triplet"]),
        _candidate("physics_health", base_architecture, heads=["health"], losses=["data", "health", "health_monotonic"], weights={"lambda_health": 0.1, "lambda_health_monotonic": 0.02}, pairs=["adjacent"]),
        _candidate("physics_regime", base_architecture, heads=[], losses=["data", "regime"], weights={"lambda_regime": 0.02}, pairs=["regime"]),
        _candidate("physics_temporal_combined", base_architecture, heads=["rate"], losses=["data", "monotonic", "rate", "smooth"], weights={"lambda_monotonic": 0.05, "lambda_rate": 0.03, "lambda_smooth": 0.02}, pairs=["adjacent", "fixed_gap", "triplet"]),
        _candidate("physics_full", base_architecture, heads=["health", "rate"], losses=["data", "monotonic", "rate", "smooth", "health", "health_monotonic", "regime", "nonnegative"], weights={"lambda_monotonic": 0.05, "lambda_rate": 0.03, "lambda_smooth": 0.02, "lambda_health": 0.1, "lambda_health_monotonic": 0.02, "lambda_regime": 0.02, "lambda_nonnegative": 0.01}, pairs=["adjacent", "fixed_gap", "triplet", "regime"]),
        _candidate("physics_full_safety", base_architecture, heads=["health", "rate"], losses=["data", "monotonic", "rate", "smooth", "health", "health_monotonic", "regime", "nonnegative", "optimistic"], weights={"lambda_monotonic": 0.05, "lambda_rate": 0.03, "lambda_smooth": 0.02, "lambda_health": 0.1, "lambda_health_monotonic": 0.02, "lambda_regime": 0.02, "lambda_nonnegative": 0.01, "lambda_optimistic": 0.05}, pairs=["adjacent", "fixed_gap", "triplet", "regime"]),
    ]
    validate_candidate_registry(candidates, max_candidates=10)
    return candidates


def validate_candidate_registry(candidates: list[dict[str, Any]], *, max_candidates: int = 10) -> None:
    if not candidates:
        raise ValueError("candidate registry must not be empty.")
    if len(candidates) > int(max_candidates):
        raise ValueError("Too many Phase 5C candidates.")
    ids = [str(candidate.get("candidate_id", "")) for candidate in candidates]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be non-empty and unique.")
    if "phase5b_reimplementation_baseline" not in ids:
        raise ValueError("The fixed Phase 5B baseline candidate is required.")
    for candidate in candidates:
        _validate_candidate(candidate)


def active_loss_weights(candidate: dict[str, Any]) -> dict[str, float]:
    weights = dict(DEFAULT_LOSS_WEIGHTS)
    weights.update(candidate.get("loss_weights", {}))
    return {key: float(value) for key, value in weights.items()}


def _candidate(
    candidate_id: str,
    architecture: dict[str, Any],
    *,
    heads: list[str],
    losses: list[str],
    weights: dict[str, float],
    pairs: list[str] | None = None,
) -> dict[str, Any]:
    all_weights = dict(DEFAULT_LOSS_WEIGHTS)
    all_weights.update(weights)
    return {
        "candidate_id": candidate_id,
        "architecture_parameters": deepcopy(architecture),
        "active_output_heads": list(heads),
        "active_losses": list(losses),
        "loss_weights": all_weights,
        "loss_tolerances": {
            "monotonic_tolerance": 1.0,
            "rate_tolerance": 2.0,
            "smoothness_tolerance": 3.0,
            "health_monotonic_tolerance": 0.02,
            "regime_tolerance": 0.05,
        },
        "pairing_requirements": list(pairs or []),
        "warm_start": {"enabled": True, "load_encoder_only": True, "strict": False},
        "training_schedule": "schedule_b",
        "parameter_budget": 1_000_000,
        "random_seed": 9701,
    }


def _validate_candidate(candidate: dict[str, Any]) -> None:
    for key in [
        "candidate_id",
        "architecture_parameters",
        "active_output_heads",
        "active_losses",
        "loss_weights",
        "loss_tolerances",
        "pairing_requirements",
        "warm_start",
        "training_schedule",
        "parameter_budget",
        "random_seed",
    ]:
        if key not in candidate:
            raise ValueError(f"Candidate is missing required field: {key}")
    weights = active_loss_weights(candidate)
    for name, value in weights.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if weights["lambda_data"] <= 0.0:
        raise ValueError("lambda_data must be greater than zero.")
    losses = set(candidate["active_losses"])
    heads = set(candidate["active_output_heads"])
    pairs = set(candidate["pairing_requirements"])
    if "health" in losses and "health" not in heads:
        raise ValueError("health loss requires the health output head.")
    if "health_monotonic" in losses and "health" not in heads:
        raise ValueError("health_monotonic loss requires the health output head.")
    if "rate" in losses and "fixed_gap" not in pairs and "adjacent" not in pairs:
        raise ValueError("rate loss requires temporal pair construction.")
    if "monotonic" in losses and "adjacent" not in pairs and "fixed_gap" not in pairs:
        raise ValueError("monotonic loss requires temporal pair construction.")
    if "smooth" in losses and "triplet" not in pairs:
        raise ValueError("smooth loss requires triplet construction.")
    if "regime" in losses and "regime" not in pairs:
        raise ValueError("regime loss requires regime-pair construction.")
    if int(candidate["parameter_budget"]) <= 0:
        raise ValueError("parameter_budget must be positive.")
    architecture = candidate["architecture_parameters"]
    if int(architecture["patch_length"]) <= 0 or int(architecture["patch_stride"]) <= 0:
        raise ValueError("Invalid patch settings.")
    if int(architecture["patch_length"]) > int(architecture["window_length"]):
        raise ValueError("patch_length must not exceed window_length.")
    if int(architecture["projection_dim"]) % int(architecture["heads"]) != 0:
        raise ValueError("projection_dim must be divisible by heads.")
