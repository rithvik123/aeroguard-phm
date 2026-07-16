"""Training-only health proxy targets for Phase 5C auxiliary heads."""

from __future__ import annotations

import numpy as np
import torch


def normalized_capped_rul_targets(capped_rul: object, rul_cap: float) -> torch.Tensor:
    """Map capped RUL to a [0, 1] proxy where higher means healthier."""

    if float(rul_cap) <= 0:
        raise ValueError("rul_cap must be positive.")
    values = torch.as_tensor(capped_rul, dtype=torch.float32).view(-1, 1)
    if not torch.isfinite(values).all():
        raise ValueError("capped_rul must be finite.")
    targets = values.clamp(min=0.0, max=float(rul_cap)) / float(rul_cap)
    validate_health_range(targets)
    return targets


def normalized_life_fraction_targets(
    cycles: object,
    failure_cycles: object,
    *,
    full_run_to_failure: bool = True,
) -> torch.Tensor:
    """Build a health proxy from cycle position in run-to-failure training engines."""

    if not full_run_to_failure:
        raise ValueError("Life-fraction health targets require full run-to-failure training engines.")
    cycle = torch.as_tensor(cycles, dtype=torch.float32).view(-1, 1)
    failure = torch.as_tensor(failure_cycles, dtype=torch.float32).view(-1, 1)
    if cycle.shape != failure.shape:
        raise ValueError("cycles and failure_cycles must have matching shapes.")
    if not torch.isfinite(cycle).all() or not torch.isfinite(failure).all():
        raise ValueError("cycles and failure_cycles must be finite.")
    if (cycle <= 0).any() or (failure <= 0).any() or (cycle > failure).any():
        raise ValueError("cycles must be positive and no greater than failure_cycles.")
    denom = (failure - 1.0).clamp_min(1.0)
    target = 1.0 - ((cycle - 1.0) / denom)
    target = target.clamp(0.0, 1.0)
    validate_health_range(target)
    return target


def validate_health_range(values: torch.Tensor) -> None:
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("health targets must have shape [n, 1].")
    if not torch.isfinite(values).all():
        raise ValueError("health targets must be finite.")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("health targets must be in [0, 1].")


def health_rul_consistency_diagnostics(
    predicted_rul: object,
    health_score: object,
    *,
    tolerance: float = 0.0,
) -> dict[str, float]:
    """Measure whether a learned health proxy orders samples like predicted RUL."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    rul = np.asarray(predicted_rul, dtype=float).reshape(-1)
    health = np.asarray(health_score, dtype=float).reshape(-1)
    if rul.shape != health.shape:
        raise ValueError("predicted_rul and health_score must be aligned.")
    if rul.size == 0:
        return {
            "health_rul_spearman": 0.0,
            "health_directional_disagreement_count": 0.0,
            "health_directional_disagreement_rate": 0.0,
        }
    if not np.isfinite(rul).all() or not np.isfinite(health).all():
        raise ValueError("health consistency inputs must be finite.")
    rul_rank = _rank(rul)
    health_rank = _rank(health)
    if np.std(rul_rank) == 0 or np.std(health_rank) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(rul_rank, health_rank)[0, 1])
    rul_diff = rul[:, None] - rul[None, :]
    health_diff = health[:, None] - health[None, :]
    upper = np.triu(np.ones_like(rul_diff, dtype=bool), k=1)
    disagreement = (rul_diff * health_diff < -float(tolerance)) & upper
    total = int(upper.sum())
    count = int(disagreement.sum())
    return {
        "health_rul_spearman": corr,
        "health_directional_disagreement_count": float(count),
        "health_directional_disagreement_rate": float(count / total) if total else 0.0,
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = float(np.mean(np.arange(start, stop)))
        start = stop
    return ranks
