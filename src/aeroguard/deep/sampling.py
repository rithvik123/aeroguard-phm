"""Deterministic engine-balanced endpoint sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aeroguard.deep.windowing import WindowSpec, candidate_endpoint_indices


def sample_engine_endpoints(
    engine_frame: pd.DataFrame,
    spec: WindowSpec,
    maximum_windows: int,
    seed: int,
) -> list[int]:
    candidates = candidate_endpoint_indices(engine_frame, spec)
    if not candidates:
        return []
    if len(candidates) <= int(maximum_windows):
        return candidates
    rng = np.random.default_rng(int(seed))
    final_index = len(engine_frame) - 1
    selected = {final_index}
    # Uniform deterministic coverage across life.
    uniform_positions = np.linspace(0, len(candidates) - 1, max(1, int(maximum_windows) // 2), dtype=int)
    selected.update(candidates[int(pos)] for pos in uniform_positions)
    # Preserve low/mid/high-RUL representation when possible.
    rul = engine_frame.iloc[candidates]["rul_capped"].to_numpy(dtype=float)
    for low, high in [(0, 15), (15, 30), (30, 60), (60, 125)]:
        band_indices = [candidates[i] for i, value in enumerate(rul) if low <= value <= high]
        if band_indices:
            selected.add(band_indices[len(band_indices) // 2])
    remaining = [idx for idx in candidates if idx not in selected]
    needed = max(0, int(maximum_windows) - len(selected))
    if needed and remaining:
        chosen = rng.choice(remaining, size=min(needed, len(remaining)), replace=False)
        selected.update(int(item) for item in chosen)
    selected_sorted = sorted(selected)
    if len(selected_sorted) > int(maximum_windows):
        if int(maximum_windows) == 1:
            return [final_index]
        pool = [idx for idx in selected_sorted if idx != final_index]
        positions = np.linspace(0, len(pool) - 1, int(maximum_windows) - 1, dtype=int)
        selected_sorted = sorted({final_index, *(pool[int(pos)] for pos in positions)})
    return selected_sorted


def build_endpoint_table(
    frame: pd.DataFrame,
    spec: WindowSpec,
    maximum_windows_per_engine: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for ordinal, (engine, group) in enumerate(frame.sort_values(["global_engine_id", "cycle"]).groupby("global_engine_id")):
        endpoints = sample_engine_endpoints(
            group.reset_index(drop=True),
            spec,
            int(maximum_windows_per_engine),
            int(seed) + ordinal,
        )
        for endpoint in endpoints:
            rows.append({"global_engine_id": engine, "endpoint_index": int(endpoint)})
    return pd.DataFrame(rows)
