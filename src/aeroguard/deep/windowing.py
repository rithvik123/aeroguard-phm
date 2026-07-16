"""Past-only sequence window creation for C-MAPSS engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from aeroguard.data.columns import CYCLE_COLUMN


@dataclass(frozen=True)
class WindowSpec:
    window_length: int
    stride: int
    minimum_valid_history: int

    def __post_init__(self) -> None:
        if self.window_length <= 0:
            raise ValueError("window_length must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        if not 1 <= self.minimum_valid_history <= self.window_length:
            raise ValueError("minimum_valid_history must be in [1, window_length].")


def candidate_endpoint_indices(engine_frame: pd.DataFrame, spec: WindowSpec) -> list[int]:
    n_rows = len(engine_frame)
    if n_rows < spec.minimum_valid_history:
        return []
    indices = list(range(spec.minimum_valid_history - 1, n_rows, spec.stride))
    if n_rows - 1 not in indices:
        indices.append(n_rows - 1)
    return sorted(set(indices))


def endpoints_for_normalized_positions(engine_frame: pd.DataFrame, positions: Iterable[float]) -> list[int]:
    n_rows = len(engine_frame)
    if n_rows == 0:
        return []
    indices = []
    for position in positions:
        value = float(position)
        if not 0.0 < value <= 1.0:
            raise ValueError("Normalized-life positions must be in (0, 1].")
        target_index = int(round((n_rows - 1) * value))
        indices.append(min(max(target_index, 0), n_rows - 1))
    return sorted(set(indices))


def build_window_from_endpoint(
    engine_frame: pd.DataFrame,
    feature_columns: list[str],
    endpoint_index: int,
    spec: WindowSpec,
) -> tuple[np.ndarray, int, int]:
    if endpoint_index < 0 or endpoint_index >= len(engine_frame):
        raise ValueError("endpoint_index is outside the engine frame.")
    start = max(0, endpoint_index - spec.window_length + 1)
    values = engine_frame.iloc[start : endpoint_index + 1][feature_columns].to_numpy(dtype=np.float32)
    valid_length = int(len(values))
    padded = np.zeros((spec.window_length, len(feature_columns)), dtype=np.float32)
    mask = np.zeros((spec.window_length, 1), dtype=np.float32)
    padded[-valid_length:, :] = values
    mask[-valid_length:, 0] = 1.0
    return np.concatenate([padded, mask], axis=1), valid_length, spec.window_length - valid_length


def build_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    endpoint_table: pd.DataFrame,
    spec: WindowSpec,
    target_column: str = "rul_capped",
    uncapped_column: str = "true_rul_uncapped",
    *,
    require_target: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    if require_target:
        missing_targets = [column for column in [target_column, uncapped_column] if column not in frame.columns]
        if missing_targets:
            raise ValueError(f"Training windows require target column(s): {missing_targets}")
    missing_features = [column for column in feature_columns if column not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns for sequence windows: {missing_features}")
    sequences = []
    rows = []
    engine_groups = {
        str(engine): group.sort_values(CYCLE_COLUMN).reset_index(drop=True)
        for engine, group in frame.groupby("global_engine_id", sort=False)
    }
    for _, endpoint in endpoint_table.iterrows():
        engine = str(endpoint["global_engine_id"])
        group = engine_groups[engine]
        endpoint_index = int(endpoint["endpoint_index"])
        sequence, valid_length, padded_count = build_window_from_endpoint(group, feature_columns, endpoint_index, spec)
        target_row = group.iloc[endpoint_index]
        row = {
            "subset": target_row.get("subset", endpoint.get("subset", "")),
            "source_domain": target_row.get("source_domain", target_row.get("subset", "")),
            "global_engine_id": target_row["global_engine_id"],
            "local_unit_id": int(target_row["local_unit_id"]) if "local_unit_id" in target_row else int(target_row.get("unit_id", -1)),
            "unit_id": int(target_row["unit_id"]),
            "cycle": int(target_row[CYCLE_COLUMN]),
            "endpoint_index": endpoint_index,
            "endpoint_cycle": int(target_row[CYCLE_COLUMN]),
            "sequence_valid_length": valid_length,
            "padded_cycle_count": padded_count,
            "operating_regime": int(target_row["operating_regime"]) if "operating_regime" in target_row else -1,
            "proxy_health_region": target_row.get("proxy_health_region", ""),
        }
        if require_target:
            row["target_rul_capped"] = float(target_row[target_column])
            row["target_rul_uncapped"] = float(target_row[uncapped_column])
        sequences.append(sequence)
        rows.append(row)
    if not sequences:
        return np.empty((0, spec.window_length, len(feature_columns) + 1), dtype=np.float32), pd.DataFrame(rows)
    return np.stack(sequences).astype(np.float32), pd.DataFrame(rows)


def build_training_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    endpoint_table: pd.DataFrame,
    spec: WindowSpec,
    target_column: str = "rul_capped",
    uncapped_column: str = "true_rul_uncapped",
) -> tuple[np.ndarray, pd.DataFrame]:
    return build_windows(
        frame,
        feature_columns,
        endpoint_table,
        spec,
        target_column=target_column,
        uncapped_column=uncapped_column,
        require_target=True,
    )


def build_inference_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    endpoint_table: pd.DataFrame,
    spec: WindowSpec,
) -> tuple[np.ndarray, pd.DataFrame]:
    return build_windows(frame, feature_columns, endpoint_table, spec, require_target=False)


def final_endpoint_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for engine, group in frame.sort_values(["global_engine_id", CYCLE_COLUMN]).groupby("global_engine_id"):
        rows.append({"global_engine_id": engine, "endpoint_index": len(group) - 1})
    return pd.DataFrame(rows)


def sequence_audit(frame: pd.DataFrame, endpoint_table: pd.DataFrame, spec: WindowSpec, rul_bands: list[dict[str, object]]) -> pd.DataFrame:
    sampled_counts = endpoint_table.groupby("global_engine_id").size().to_dict() if not endpoint_table.empty else {}
    rows = []
    for engine, group in frame.sort_values(["global_engine_id", CYCLE_COLUMN]).groupby("global_engine_id"):
        candidates = candidate_endpoint_indices(group, spec)
        endpoints = endpoint_table[endpoint_table["global_engine_id"] == engine]["endpoint_index"].astype(int).tolist()
        selected = group.iloc[endpoints] if endpoints else group.iloc[[]]
        row = {
            "subset": group["subset"].iloc[0],
            "global_engine_id": engine,
            "engine_length": int(len(group)),
            "candidate_window_count": int(len(candidates)),
            "sampled_window_count": int(sampled_counts.get(engine, 0)),
            "padded_window_count": int(sum(index + 1 < spec.window_length for index in endpoints)),
            "minimum_target_rul": None if selected.empty else float(selected["true_rul_uncapped"].min()),
            "maximum_target_rul": None if selected.empty else float(selected["true_rul_uncapped"].max()),
        }
        for band in rul_bands:
            lower = float(band["lower"])
            upper = band.get("upper")
            upper_value = np.inf if upper is None else float(upper)
            label = str(band["label"])
            row[f"rul_band_{label}_count"] = int(((selected["true_rul_uncapped"] >= lower) & (selected["true_rul_uncapped"] <= upper_value)).sum()) if not selected.empty else 0
        rows.append(row)
    return pd.DataFrame(rows)
