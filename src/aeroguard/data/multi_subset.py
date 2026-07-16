"""Multi-subset C-MAPSS loading with collision-safe engine identities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from aeroguard.data.columns import CYCLE_COLUMN, UNIT_COLUMN
from aeroguard.data.loader import load_cmapss_dataset
from aeroguard.data.targets import add_training_rul_targets
from aeroguard.onset.onset_detection import add_proxy_labels, derive_test_true_rul_trajectory


def add_subset_identity(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Add subset, source-domain, local ID, and collision-safe global engine ID."""
    subset = subset.upper()
    result = frame.copy()
    result["subset"] = subset
    result["source_domain"] = subset
    result["local_unit_id"] = result[UNIT_COLUMN].astype(int)
    result["global_engine_id"] = result["local_unit_id"].map(lambda value: f"{subset}_{int(value):04d}")
    return result


def validate_global_engine_index(frame: pd.DataFrame) -> None:
    """Validate global engine IDs and per-engine cycle uniqueness."""
    if frame["global_engine_id"].isna().any():
        raise ValueError("global_engine_id contains missing values.")
    pairs = frame[["global_engine_id", CYCLE_COLUMN]]
    if pairs.duplicated().any():
        duplicate = pairs[pairs.duplicated()].iloc[0].to_dict()
        raise ValueError(f"Duplicate global-engine-cycle pair detected: {duplicate}")
    mapping = frame[["subset", "local_unit_id", "global_engine_id"]].drop_duplicates()
    if mapping["global_engine_id"].duplicated().any():
        raise ValueError("global_engine_id collision detected.")


def load_training_subsets(
    dataset_dir: str | Path,
    subsets: Iterable[str],
    rul_cap: float,
    healthy_rul_threshold: float,
    critical_rul_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and concatenate C-MAPSS training subsets with RUL targets."""
    pieces = []
    metadata: dict[str, object] = {"subsets": {}}
    for subset in [str(item).upper() for item in subsets]:
        dataset = load_cmapss_dataset(dataset_dir, subset)
        train = add_training_rul_targets(dataset.train, rul_cap=float(rul_cap))
        train["true_rul_uncapped"] = train["rul_uncapped"]
        train = add_proxy_labels(
            train,
            healthy_rul_threshold=float(healthy_rul_threshold),
            critical_rul_threshold=float(critical_rul_threshold),
        )
        train = add_subset_identity(train, subset)
        pieces.append(train)
        metadata["subsets"][subset] = {
            "train_shape": list(dataset.train.shape),
            "train_engine_count": int(dataset.train[UNIT_COLUMN].nunique()),
            "train_file": str(dataset.files.train),
        }
    combined = pd.concat(pieces, ignore_index=True)
    validate_global_engine_index(combined)
    metadata["combined_shape"] = list(combined.shape)
    metadata["combined_engine_count"] = int(combined["global_engine_id"].nunique())
    return combined, metadata


def load_test_subset(
    dataset_dir: str | Path,
    subset: str,
    healthy_rul_threshold: float,
    critical_rul_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load one C-MAPSS test subset and derive per-cycle true RUL."""
    subset = subset.upper()
    dataset = load_cmapss_dataset(dataset_dir, subset)
    test = derive_test_true_rul_trajectory(dataset.test, dataset.test_rul)
    test = add_proxy_labels(
        test,
        healthy_rul_threshold=float(healthy_rul_threshold),
        critical_rul_threshold=float(critical_rul_threshold),
    )
    test = add_subset_identity(test, subset)
    validate_global_engine_index(test)
    final_rows = test.sort_values(["global_engine_id", CYCLE_COLUMN]).groupby("global_engine_id").tail(1)
    if len(final_rows) != len(dataset.test_rul):
        raise ValueError(f"RUL row count mismatch for {subset}.")
    metadata = {
        "subset": subset,
        "test_shape": list(dataset.test.shape),
        "test_engine_count": int(dataset.test[UNIT_COLUMN].nunique()),
        "test_file": str(dataset.files.test),
        "rul_file": str(dataset.files.rul),
    }
    return test, metadata


def load_test_subsets(
    dataset_dir: str | Path,
    subsets: Iterable[str],
    healthy_rul_threshold: float,
    critical_rul_threshold: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Load several test subsets keyed by subset name."""
    frames: dict[str, pd.DataFrame] = {}
    metadata: dict[str, object] = {}
    for subset in [str(item).upper() for item in subsets]:
        frame, meta = load_test_subset(dataset_dir, subset, healthy_rul_threshold, critical_rul_threshold)
        frames[subset] = frame
        metadata[subset] = meta
    return frames, metadata
