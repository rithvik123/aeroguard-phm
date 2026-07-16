"""Deterministic same-engine temporal pair construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_METADATA_COLUMNS = {
    "subset",
    "global_engine_id",
    "cycle",
    "target_rul_capped",
    "operating_regime",
    "sequence_valid_length",
}


@dataclass(frozen=True)
class TemporalPairingConfig:
    adjacent_enabled: bool = True
    fixed_gap_enabled: bool = True
    triplet_enabled: bool = True
    allowed_cycle_gaps: tuple[int, ...] = (1, 5)
    max_adjacent_pairs_per_engine: int = 16
    max_fixed_gap_pairs_per_engine: int = 16
    max_triplets_per_engine: int = 8
    seed: int = 1701
    sampling_method: str = "uniform"


def build_temporal_pairs(metadata: pd.DataFrame, config: TemporalPairingConfig) -> pd.DataFrame:
    """Build adjacent, fixed-gap, and triplet records without crossing engines."""

    _validate_config(config)
    frame = _metadata_frame(metadata)
    rows: list[dict[str, object]] = []
    for (subset, engine), group in frame.groupby(["subset", "global_engine_id"], sort=True):
        group = group.sort_values("cycle").reset_index(drop=True)
        by_cycle = {int(row["cycle"]): row for _, row in group.iterrows()}
        adjacent: list[dict[str, object]] = []
        fixed: list[dict[str, object]] = []
        triplets: list[dict[str, object]] = []
        if config.adjacent_enabled:
            for cycle in sorted(by_cycle):
                if cycle + 1 in by_cycle:
                    adjacent.append(_pair_row(by_cycle[cycle], by_cycle[cycle + 1], "adjacent", 1))
        if config.fixed_gap_enabled:
            for gap in config.allowed_cycle_gaps:
                if gap <= 1:
                    continue
                for cycle in sorted(by_cycle):
                    if cycle + gap in by_cycle:
                        fixed.append(_pair_row(by_cycle[cycle], by_cycle[cycle + gap], "fixed_gap", gap))
        if config.triplet_enabled:
            for gap in config.allowed_cycle_gaps:
                for cycle in sorted(by_cycle):
                    if cycle - gap in by_cycle and cycle + gap in by_cycle:
                        earlier = by_cycle[cycle - gap]
                        middle = by_cycle[cycle]
                        later = by_cycle[cycle + gap]
                        triplets.append(_triplet_row(earlier, middle, later, gap))
        rows.extend(_bounded(adjacent, config.max_adjacent_pairs_per_engine, config.seed, config.sampling_method, f"{subset}-{engine}-adjacent"))
        rows.extend(_bounded(fixed, config.max_fixed_gap_pairs_per_engine, config.seed, config.sampling_method, f"{subset}-{engine}-fixed"))
        rows.extend(_bounded(triplets, config.max_triplets_per_engine, config.seed, config.sampling_method, f"{subset}-{engine}-triplet"))
    return pd.DataFrame(rows)


def pair_indices(pair_frame: pd.DataFrame, pair_type: str | Iterable[str] = ("adjacent", "fixed_gap")) -> np.ndarray:
    """Return [n, 2] index pairs for temporal losses."""

    if pair_frame.empty:
        return np.empty((0, 2), dtype=np.int64)
    types = {pair_type} if isinstance(pair_type, str) else set(pair_type)
    subset = pair_frame[pair_frame["pair_type"].isin(types)]
    if subset.empty:
        return np.empty((0, 2), dtype=np.int64)
    return subset[["earlier_index", "later_index"]].to_numpy(dtype=np.int64)


def triplet_indices(pair_frame: pd.DataFrame) -> np.ndarray:
    if pair_frame.empty:
        return np.empty((0, 3), dtype=np.int64)
    subset = pair_frame[pair_frame["pair_type"] == "triplet"]
    if subset.empty:
        return np.empty((0, 3), dtype=np.int64)
    return subset[["earlier_index", "middle_index", "later_index"]].to_numpy(dtype=np.int64)


def _metadata_frame(metadata: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_METADATA_COLUMNS - set(metadata.columns))
    if missing:
        raise ValueError(f"Missing metadata columns for temporal pairing: {missing}")
    frame = metadata.copy()
    if "sample_index" not in frame.columns:
        frame["sample_index"] = np.arange(len(frame), dtype=np.int64)
    if frame["sample_index"].duplicated().any():
        raise ValueError("sample_index values must be unique.")
    if frame["sequence_valid_length"].astype(float).le(0).any():
        frame = frame[frame["sequence_valid_length"].astype(float) > 0].copy()
    if frame.empty:
        return frame
    if not np.isfinite(frame["cycle"].astype(float)).all() or not np.isfinite(frame["target_rul_capped"].astype(float)).all():
        raise ValueError("cycle and target_rul_capped must be finite.")
    return frame


def _validate_config(config: TemporalPairingConfig) -> None:
    if not config.allowed_cycle_gaps:
        raise ValueError("allowed_cycle_gaps must not be empty.")
    if any(int(gap) <= 0 for gap in config.allowed_cycle_gaps):
        raise ValueError("allowed_cycle_gaps must be positive.")
    for value in [config.max_adjacent_pairs_per_engine, config.max_fixed_gap_pairs_per_engine, config.max_triplets_per_engine]:
        if int(value) < 0:
            raise ValueError("Maximum pair counts must be non-negative.")
    if config.sampling_method not in {"first", "uniform"}:
        raise ValueError("sampling_method must be 'first' or 'uniform'.")


def _pair_row(earlier: pd.Series, later: pd.Series, pair_type: str, gap: int) -> dict[str, object]:
    if earlier["global_engine_id"] != later["global_engine_id"] or earlier["subset"] != later["subset"]:
        raise ValueError("Temporal pairs must stay within one engine and subset.")
    earlier_cycle = int(earlier["cycle"])
    later_cycle = int(later["cycle"])
    if later_cycle <= earlier_cycle:
        raise ValueError("Temporal pair later cycle must be after earlier cycle.")
    return {
        "pair_type": pair_type,
        "subset": earlier["subset"],
        "global_engine_id": earlier["global_engine_id"],
        "earlier_index": int(earlier["sample_index"]),
        "later_index": int(later["sample_index"]),
        "middle_index": -1,
        "earlier_cycle": earlier_cycle,
        "later_cycle": later_cycle,
        "middle_cycle": -1,
        "cycle_gap": int(gap),
        "left_gap": int(gap),
        "right_gap": int(gap),
        "earlier_true_capped_rul": float(earlier["target_rul_capped"]),
        "later_true_capped_rul": float(later["target_rul_capped"]),
        "middle_true_capped_rul": np.nan,
        "earlier_operating_regime": int(earlier["operating_regime"]),
        "later_operating_regime": int(later["operating_regime"]),
        "middle_operating_regime": -1,
        "validity_mask": True,
    }


def _triplet_row(earlier: pd.Series, middle: pd.Series, later: pd.Series, gap: int) -> dict[str, object]:
    if len({earlier["global_engine_id"], middle["global_engine_id"], later["global_engine_id"]}) != 1:
        raise ValueError("Temporal triplets must stay within one engine.")
    if len({earlier["subset"], middle["subset"], later["subset"]}) != 1:
        raise ValueError("Temporal triplets must stay within one subset.")
    return {
        "pair_type": "triplet",
        "subset": earlier["subset"],
        "global_engine_id": earlier["global_engine_id"],
        "earlier_index": int(earlier["sample_index"]),
        "middle_index": int(middle["sample_index"]),
        "later_index": int(later["sample_index"]),
        "earlier_cycle": int(earlier["cycle"]),
        "middle_cycle": int(middle["cycle"]),
        "later_cycle": int(later["cycle"]),
        "cycle_gap": int(gap),
        "left_gap": int(int(middle["cycle"]) - int(earlier["cycle"])),
        "right_gap": int(int(later["cycle"]) - int(middle["cycle"])),
        "earlier_true_capped_rul": float(earlier["target_rul_capped"]),
        "middle_true_capped_rul": float(middle["target_rul_capped"]),
        "later_true_capped_rul": float(later["target_rul_capped"]),
        "earlier_operating_regime": int(earlier["operating_regime"]),
        "middle_operating_regime": int(middle["operating_regime"]),
        "later_operating_regime": int(later["operating_regime"]),
        "validity_mask": True,
    }


def _bounded(rows: list[dict[str, object]], limit: int, seed: int, method: str, salt: str) -> list[dict[str, object]]:
    limit = int(limit)
    if limit == 0 or not rows:
        return []
    ordered = sorted(rows, key=lambda row: (int(row["earlier_cycle"]), int(row["later_cycle"]), int(row.get("middle_cycle", -1))))
    if len(ordered) <= limit:
        return ordered
    if method == "first":
        return ordered[:limit]
    rng = np.random.default_rng(int(seed) + _stable_offset(salt))
    selected = np.sort(rng.choice(len(ordered), size=limit, replace=False))
    return [ordered[int(index)] for index in selected]


def _stable_offset(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % 1_000_003
