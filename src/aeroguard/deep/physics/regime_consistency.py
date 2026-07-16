"""Operating-regime consistency helpers for physics-guided training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


REGIME_PAIR_COLUMNS = [
    "left_index",
    "right_index",
    "left_global_engine_id",
    "right_global_engine_id",
    "left_subset",
    "right_subset",
    "left_operating_regime",
    "right_operating_regime",
    "left_target_rul",
    "right_target_rul",
    "target_rul_difference",
    "left_valid_length",
    "right_valid_length",
]


@dataclass(frozen=True)
class RegimePairingConfig:
    enabled: bool = False
    rul_tolerance: float = 5.0
    max_pairs: int = 20_000
    seed: int = 2701
    sampling_method: str = "uniform"
    max_anchors: int = 10_000
    max_partners_per_anchor: int = 2
    max_pairs_per_regime_pair: int = 4_000
    allow_empty_pairs: bool = True
    lazy_build: bool = True
    cache_bounded_pairs: bool = True


def empty_regime_pair_frame(reason: str = "disabled") -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "left_index": pd.Series(dtype="int32"),
            "right_index": pd.Series(dtype="int32"),
            "left_global_engine_id": pd.Series(pd.Categorical([])),
            "right_global_engine_id": pd.Series(pd.Categorical([])),
            "left_subset": pd.Series(pd.Categorical([])),
            "right_subset": pd.Series(pd.Categorical([])),
            "left_operating_regime": pd.Series(dtype="int16"),
            "right_operating_regime": pd.Series(dtype="int16"),
            "left_target_rul": pd.Series(dtype="float32"),
            "right_target_rul": pd.Series(dtype="float32"),
            "target_rul_difference": pd.Series(dtype="float32"),
            "left_valid_length": pd.Series(dtype="int32"),
            "right_valid_length": pd.Series(dtype="int32"),
        }
    )
    frame.attrs["diagnostics"] = _empty_diagnostics(reason)
    return frame


def build_regime_pairs(metadata: pd.DataFrame, config: RegimePairingConfig) -> pd.DataFrame:
    """Build bounded cross-regime pairs without materializing all candidates."""

    start = time.perf_counter()
    if not config.enabled:
        return empty_regime_pair_frame("disabled")
    _validate_config(config)
    frame = _compact_metadata(metadata)
    diagnostics = _base_diagnostics(metadata, frame, config)
    if frame.empty:
        if config.allow_empty_pairs:
            return _with_diagnostics(empty_regime_pair_frame("no_valid_samples"), diagnostics, start)
        raise ValueError("No valid samples are available for regime pairing.")
    regimes = np.asarray(sorted(frame["operating_regime"].unique()), dtype=np.int32)
    diagnostics["number_of_regimes"] = int(len(regimes))
    if len(regimes) < 2:
        if config.allow_empty_pairs:
            return _with_diagnostics(empty_regime_pair_frame("only_one_regime"), diagnostics, start)
        raise ValueError("Regime consistency requires at least two operating regimes.")

    ordered = frame.sort_values(["target_rul_capped", "operating_regime", "sample_index"], kind="mergesort").reset_index(drop=True)
    arrays = _metadata_arrays(ordered)
    regime_positions = {
        int(regime): np.flatnonzero(arrays["regime"] == int(regime)).astype(np.int32)
        for regime in regimes
    }
    anchors = _select_anchors(len(ordered), int(config.max_anchors), int(config.seed), str(config.sampling_method))
    diagnostics["anchor_count_considered"] = int(len(anchors))
    pair_keys: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    regime_pair_counts: dict[tuple[int, int], int] = {}
    matches_examined = 0
    global_limit_reached = False
    per_combination_limit_reached = False
    partner_limit_reached = False
    rng = np.random.default_rng(_stable_seed("regime-partners", int(config.seed)))

    for anchor_position in anchors:
        if len(rows) >= int(config.max_pairs):
            global_limit_reached = True
            break
        anchor_regime = int(arrays["regime"][anchor_position])
        anchor_rul = float(arrays["rul"][anchor_position])
        partners_for_anchor = 0
        other_regimes = [int(regime) for regime in regimes if int(regime) != anchor_regime]
        if str(config.sampling_method) == "uniform":
            other_regimes = _deterministic_shuffle(other_regimes, int(config.seed), int(arrays["sample_index"][anchor_position]))
        for other_regime in other_regimes:
            if partners_for_anchor >= int(config.max_partners_per_anchor):
                partner_limit_reached = True
                break
            combo = (anchor_regime, other_regime)
            if regime_pair_counts.get(combo, 0) >= int(config.max_pairs_per_regime_pair):
                per_combination_limit_reached = True
                continue
            positions = regime_positions[other_regime]
            if len(positions) == 0:
                continue
            other_rul = arrays["rul"][positions]
            lower = np.searchsorted(other_rul, anchor_rul - float(config.rul_tolerance), side="left")
            upper = np.searchsorted(other_rul, anchor_rul + float(config.rul_tolerance), side="right")
            if upper <= lower:
                continue
            candidates = positions[lower:upper]
            candidates = candidates[candidates != anchor_position]
            if candidates.size == 0:
                continue
            matches_examined += int(candidates.size)
            candidate_order = _candidate_order(candidates, arrays["sample_index"][anchor_position], rng, str(config.sampling_method))
            for partner_position in candidate_order:
                if partners_for_anchor >= int(config.max_partners_per_anchor):
                    partner_limit_reached = True
                    break
                if len(rows) >= int(config.max_pairs):
                    global_limit_reached = True
                    break
                if regime_pair_counts.get(combo, 0) >= int(config.max_pairs_per_regime_pair):
                    per_combination_limit_reached = True
                    break
                left_sample = int(arrays["sample_index"][anchor_position])
                right_sample = int(arrays["sample_index"][partner_position])
                if left_sample == right_sample:
                    continue
                key = tuple(sorted((left_sample, right_sample)))
                if key in pair_keys:
                    continue
                pair_keys.add(key)
                rows.append(_pair_row(arrays, anchor_position, int(partner_position)))
                regime_pair_counts[combo] = regime_pair_counts.get(combo, 0) + 1
                partners_for_anchor += 1
        if global_limit_reached:
            break

    diagnostics.update(
        {
            "candidate_matches_examined": int(matches_examined),
            "retained_pair_count": int(len(rows)),
            "global_pair_limit": int(config.max_pairs),
            "global_limit_reached": bool(global_limit_reached or len(rows) >= int(config.max_pairs)),
            "per_anchor_limit_reached": bool(partner_limit_reached),
            "per_regime_combination_limit_reached": bool(per_combination_limit_reached),
            "regime_pair_limit_counts": {f"{left}->{right}": int(count) for (left, right), count in sorted(regime_pair_counts.items())},
            "empty_reason": "" if rows else "no_cross_regime_rul_matches",
        }
    )
    if not rows:
        if config.allow_empty_pairs:
            return _with_diagnostics(empty_regime_pair_frame("no_cross_regime_rul_matches"), diagnostics, start)
        raise ValueError("No cross-regime samples satisfy the configured RUL tolerance.")
    result = pd.DataFrame(rows, columns=REGIME_PAIR_COLUMNS)
    result = _compact_pair_frame(result)
    return _with_diagnostics(result, diagnostics, start)


def regime_pair_memory_diagnostics(metadata: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, float | int]:
    return {
        "metadata_rows": int(len(metadata)),
        "metadata_columns": int(len(metadata.columns)),
        "metadata_memory_mb": _memory_mb(metadata),
        "pair_rows": int(len(pairs)),
        "pair_columns": int(len(pairs.columns)),
        "pair_memory_mb": _memory_mb(pairs),
    }


def latent_consistency_loss(left_latent: torch.Tensor, right_latent: torch.Tensor, *, metric: str = "cosine", tolerance: float = 0.0) -> torch.Tensor:
    if left_latent.ndim != 2 or right_latent.ndim != 2 or left_latent.shape != right_latent.shape:
        raise ValueError("Latent tensors must have matching shape [n, features].")
    if not torch.isfinite(left_latent).all() or not torch.isfinite(right_latent).all():
        raise ValueError("Latent tensors must be finite.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    if left_latent.numel() == 0:
        return left_latent.sum() * 0.0
    if metric == "cosine":
        distance = 1.0 - F.cosine_similarity(left_latent, right_latent, dim=1)
    elif metric == "l2":
        distance = torch.linalg.vector_norm(left_latent - right_latent, dim=1)
    else:
        raise ValueError(f"Unsupported latent distance metric: {metric}")
    return torch.relu(distance - float(tolerance)).mean()


def prediction_consistency_loss(left_prediction: torch.Tensor, right_prediction: torch.Tensor, *, tolerance: float = 0.0) -> torch.Tensor:
    if left_prediction.ndim == 1:
        left_prediction = left_prediction.view(-1, 1)
    if right_prediction.ndim == 1:
        right_prediction = right_prediction.view(-1, 1)
    if left_prediction.shape != right_prediction.shape or left_prediction.ndim != 2 or left_prediction.shape[1] != 1:
        raise ValueError("Prediction tensors must have matching shape [n, 1].")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    return torch.relu((left_prediction - right_prediction).abs() - float(tolerance)).mean()


def regime_pair_diagnostics(
    left_latent: torch.Tensor | None,
    right_latent: torch.Tensor | None,
    left_prediction: torch.Tensor,
    right_prediction: torch.Tensor,
    *,
    target_rul_difference: object | None = None,
    tolerance: float = 0.0,
) -> dict[str, float]:
    pred_gap = (left_prediction.detach().view(-1) - right_prediction.detach().view(-1)).abs()
    if not torch.isfinite(pred_gap).all():
        raise ValueError("Prediction disagreement values must be finite.")
    latent_distance = torch.zeros_like(pred_gap)
    if left_latent is not None and right_latent is not None and left_latent.numel():
        latent_distance = torch.linalg.vector_norm(
            F.normalize(left_latent.detach(), dim=1) - F.normalize(right_latent.detach(), dim=1),
            dim=1,
        )
    target_diff = np.asarray(target_rul_difference if target_rul_difference is not None else [], dtype=float).reshape(-1)
    return {
        "regime_pair_count": float(pred_gap.numel()),
        "regime_mean_target_rul_difference": float(target_diff.mean()) if target_diff.size else 0.0,
        "regime_mean_latent_distance": float(latent_distance.mean().item()) if latent_distance.numel() else 0.0,
        "regime_mean_prediction_disagreement": float(pred_gap.mean().item()) if pred_gap.numel() else 0.0,
        "regime_consistency_violation_rate": float((pred_gap > float(tolerance)).to(dtype=torch.float32).mean().item()) if pred_gap.numel() else 0.0,
    }


def _compact_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_index", "target_rul_capped", "operating_regime", "sequence_valid_length"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Missing metadata columns for regime pairing: {missing}")
    if "data_role" in metadata.columns and metadata["data_role"].astype(str).str.contains("benchmark|test", case=False, regex=True).any():
        raise ValueError("Benchmark/test rows cannot enter regime-consistency training pairs.")
    if "subset" in metadata.columns and metadata["subset"].astype(str).str.lower().str.startswith("test_").any():
        raise ValueError("Benchmark/test subset rows cannot enter regime-consistency training pairs.")
    frame = pd.DataFrame(
        {
            "sample_index": metadata["sample_index"],
            "target_rul_capped": metadata["target_rul_capped"],
            "operating_regime": metadata["operating_regime"],
            "sequence_valid_length": metadata["sequence_valid_length"],
            "global_engine_id": metadata["global_engine_id"] if "global_engine_id" in metadata.columns else "",
            "subset": metadata["subset"] if "subset" in metadata.columns else "",
        }
    )
    frame = frame[frame["sequence_valid_length"].astype(float) > 0].copy()
    if frame.empty:
        return frame
    if not np.isfinite(frame[["sample_index", "target_rul_capped", "operating_regime", "sequence_valid_length"]].to_numpy(dtype=float)).all():
        raise ValueError("Regime-pair metadata numeric fields must be finite.")
    frame["sample_index"] = _safe_int_series(frame["sample_index"], "sample_index")
    frame["operating_regime"] = _safe_int_series(frame["operating_regime"], "operating_regime").astype("int16")
    frame["sequence_valid_length"] = _safe_int_series(frame["sequence_valid_length"], "sequence_valid_length")
    frame["target_rul_capped"] = frame["target_rul_capped"].astype("float32")
    frame["global_engine_id"] = frame["global_engine_id"].astype(str)
    frame["subset"] = frame["subset"].astype(str)
    return frame


def _metadata_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "sample_index": frame["sample_index"].to_numpy(dtype=np.int32, copy=False),
        "rul": frame["target_rul_capped"].to_numpy(dtype=np.float32, copy=False),
        "regime": frame["operating_regime"].to_numpy(dtype=np.int16, copy=False),
        "valid_length": frame["sequence_valid_length"].to_numpy(dtype=np.int32, copy=False),
        "global_engine_id": frame["global_engine_id"].to_numpy(dtype=object, copy=False),
        "subset": frame["subset"].to_numpy(dtype=object, copy=False),
    }


def _pair_row(arrays: dict[str, np.ndarray], left_position: int, right_position: int) -> dict[str, Any]:
    return {
        "left_index": int(arrays["sample_index"][left_position]),
        "right_index": int(arrays["sample_index"][right_position]),
        "left_global_engine_id": str(arrays["global_engine_id"][left_position]),
        "right_global_engine_id": str(arrays["global_engine_id"][right_position]),
        "left_subset": str(arrays["subset"][left_position]),
        "right_subset": str(arrays["subset"][right_position]),
        "left_operating_regime": int(arrays["regime"][left_position]),
        "right_operating_regime": int(arrays["regime"][right_position]),
        "left_target_rul": float(arrays["rul"][left_position]),
        "right_target_rul": float(arrays["rul"][right_position]),
        "target_rul_difference": abs(float(arrays["rul"][right_position]) - float(arrays["rul"][left_position])),
        "left_valid_length": int(arrays["valid_length"][left_position]),
        "right_valid_length": int(arrays["valid_length"][right_position]),
    }


def _compact_pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ["left_index", "right_index", "left_valid_length", "right_valid_length"]:
        frame[column] = _safe_int_series(frame[column], column)
    for column in ["left_operating_regime", "right_operating_regime"]:
        frame[column] = _safe_int_series(frame[column], column).astype("int16")
    for column in ["left_target_rul", "right_target_rul", "target_rul_difference"]:
        frame[column] = frame[column].astype("float32")
    for column in ["left_global_engine_id", "right_global_engine_id", "left_subset", "right_subset"]:
        frame[column] = frame[column].astype("category")
    return frame.sort_values(["target_rul_difference", "left_index", "right_index"], kind="mergesort").reset_index(drop=True)


def _safe_int_series(values: pd.Series, name: str) -> pd.Series:
    maximum = float(values.max()) if len(values) else 0.0
    minimum = float(values.min()) if len(values) else 0.0
    if minimum < np.iinfo(np.int32).min or maximum > np.iinfo(np.int32).max:
        return values.astype("int64")
    return values.astype("int32")


def _select_anchors(row_count: int, limit: int, seed: int, method: str) -> np.ndarray:
    if row_count <= 0 or limit <= 0:
        return np.empty(0, dtype=np.int32)
    if row_count <= limit:
        return np.arange(row_count, dtype=np.int32)
    if method == "first":
        return np.arange(limit, dtype=np.int32)
    rng = np.random.default_rng(_stable_seed("regime-anchors", seed))
    return np.sort(rng.choice(row_count, size=limit, replace=False)).astype(np.int32)


def _candidate_order(candidates: np.ndarray, anchor_sample_index: int, rng: np.random.Generator, method: str) -> np.ndarray:
    if method == "first" or len(candidates) <= 1:
        return candidates
    order_seed = _stable_seed(f"anchor-{anchor_sample_index}", int(rng.integers(0, 2**31 - 1)))
    local_rng = np.random.default_rng(order_seed)
    return candidates[local_rng.permutation(len(candidates))]


def _deterministic_shuffle(values: list[int], seed: int, salt: int) -> list[int]:
    rng = np.random.default_rng(_stable_seed(f"regimes-{salt}", seed))
    values = list(values)
    if values:
        order = rng.permutation(len(values))
        values = [values[int(index)] for index in order]
    return values


def _stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{value}-{seed}".encode("utf-8")).hexdigest()[:8]
    return (int(seed) + int(digest, 16)) % (2**32 - 1)


def _base_diagnostics(original: pd.DataFrame, compact: pd.DataFrame, config: RegimePairingConfig) -> dict[str, Any]:
    return {
        "algorithm": "bounded_rul_searchsorted",
        "metadata_rows": int(len(original)),
        "metadata_columns": int(len(original.columns)),
        "metadata_memory_mb": _memory_mb(original),
        "compact_metadata_memory_mb": _memory_mb(compact),
        "number_of_regimes": int(compact["operating_regime"].nunique()) if not compact.empty else 0,
        "anchor_limit": int(config.max_anchors),
        "partners_per_anchor_limit": int(config.max_partners_per_anchor),
        "pairs_per_regime_pair_limit": int(config.max_pairs_per_regime_pair),
        "global_pair_limit": int(config.max_pairs),
        "candidate_matches_examined": 0,
        "retained_pair_count": 0,
        "limit_reached": False,
    }


def _with_diagnostics(frame: pd.DataFrame, diagnostics: dict[str, Any], start: float) -> pd.DataFrame:
    empty_reason = dict(frame.attrs.get("diagnostics", {})).get("empty_reason", "")
    diagnostics = dict(diagnostics)
    diagnostics.setdefault("empty_reason", empty_reason)
    diagnostics["retained_pair_count"] = int(len(frame))
    diagnostics["pair_table_memory_mb"] = _memory_mb(frame)
    diagnostics["runtime_seconds"] = float(time.perf_counter() - start)
    diagnostics["limit_reached"] = bool(
        diagnostics.get("global_limit_reached", False)
        or diagnostics.get("per_anchor_limit_reached", False)
        or diagnostics.get("per_regime_combination_limit_reached", False)
    )
    frame.attrs["diagnostics"] = diagnostics
    return frame


def _empty_diagnostics(reason: str) -> dict[str, Any]:
    return {
        "algorithm": "bounded_rul_searchsorted",
        "metadata_rows": 0,
        "metadata_columns": 0,
        "metadata_memory_mb": 0.0,
        "number_of_regimes": 0,
        "anchor_count_considered": 0,
        "candidate_matches_examined": 0,
        "retained_pair_count": 0,
        "pair_table_memory_mb": 0.0,
        "global_pair_limit": 0,
        "limit_reached": False,
        "empty_reason": reason,
        "runtime_seconds": 0.0,
    }


def _memory_mb(frame: pd.DataFrame) -> float:
    return float(frame.memory_usage(deep=True).sum() / (1024.0 * 1024.0))


def _validate_config(config: RegimePairingConfig) -> None:
    if float(config.rul_tolerance) < 0:
        raise ValueError("rul_tolerance must be non-negative.")
    for name, value in [
        ("max_pairs", config.max_pairs),
        ("max_anchors", config.max_anchors),
        ("max_partners_per_anchor", config.max_partners_per_anchor),
        ("max_pairs_per_regime_pair", config.max_pairs_per_regime_pair),
    ]:
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative.")
    if int(config.max_pairs) == 0 and not config.allow_empty_pairs:
        raise ValueError("max_pairs must be positive when empty pairs are not allowed.")
    if config.sampling_method not in {"first", "uniform"}:
        raise ValueError("sampling_method must be 'first' or 'uniform'.")
