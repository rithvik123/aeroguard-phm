"""Whitespace-safe NASA C-MAPSS loading functions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from aeroguard.data.columns import CMAPSS_COLUMNS, TEST_TARGET_COLUMN
from aeroguard.data.validation import (
    CMapssValidationError,
    validate_cmapss_frame,
    validate_raw_column_count,
    validate_rul_values,
    validate_test_rul_alignment,
)


SUPPORTED_SUBSETS = {"FD001", "FD002", "FD003", "FD004"}


@dataclass(frozen=True)
class CMapssFiles:
    """Resolved C-MAPSS file paths for one subset."""

    train: Path
    test: Path
    rul: Path


@dataclass(frozen=True)
class CMapssDataset:
    """Loaded C-MAPSS train/test/RUL tables."""

    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.Series
    files: CMapssFiles


def resolve_cmapss_files(dataset_dir: str | Path, subset: str) -> CMapssFiles:
    """Resolve and validate the required files for a C-MAPSS subset."""
    subset = subset.upper()
    if subset not in SUPPORTED_SUBSETS:
        raise CMapssValidationError(
            f"Unsupported C-MAPSS subset '{subset}'. Supported: {sorted(SUPPORTED_SUBSETS)}."
        )
    root = Path(dataset_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    files = CMapssFiles(
        train=root / f"train_{subset}.txt",
        test=root / f"test_{subset}.txt",
        rul=root / f"RUL_{subset}.txt",
    )
    for path in (files.train, files.test, files.rul):
        if not path.exists():
            raise FileNotFoundError(f"Required C-MAPSS file not found: {path}")
    return files


def _drop_trailing_empty_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop only trailing all-empty columns that can appear with loose parsing."""
    result = frame.copy()
    while result.shape[1] > 0 and result.iloc[:, -1].isna().all():
        result = result.iloc[:, :-1]
    return result


def read_cmapss_table(path: str | Path, source_name: str | None = None) -> pd.DataFrame:
    """Read and validate a standard 26-column C-MAPSS train/test table."""
    path = Path(path)
    source = source_name or path.name
    raw = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    raw = _drop_trailing_empty_columns(raw)
    validate_raw_column_count(raw.shape[1], source)
    raw.columns = CMAPSS_COLUMNS
    return validate_cmapss_frame(raw, source)


def read_rul_file(path: str | Path, source_name: str | None = None) -> pd.Series:
    """Read and validate a one-column C-MAPSS RUL file."""
    path = Path(path)
    source = source_name or path.name
    raw = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    raw = _drop_trailing_empty_columns(raw)
    if raw.shape[1] != 1:
        raise CMapssValidationError(
            f"{source} has {raw.shape[1]} meaningful columns; expected 1 RUL column."
        )
    values = validate_rul_values(raw.iloc[:, 0], source)
    values.name = TEST_TARGET_COLUMN
    return values


def load_cmapss_dataset(dataset_dir: str | Path, subset: str = "FD001") -> CMapssDataset:
    """Load train/test/RUL files and validate their structural relationship."""
    files = resolve_cmapss_files(dataset_dir, subset)
    train = read_cmapss_table(files.train, files.train.name)
    test = read_cmapss_table(files.test, files.test.name)
    test_rul = read_rul_file(files.rul, files.rul.name)
    validate_test_rul_alignment(test, test_rul)
    return CMapssDataset(train=train, test=test, test_rul=test_rul, files=files)
