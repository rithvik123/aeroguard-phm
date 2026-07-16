from pathlib import Path

import pytest

from aeroguard.data.columns import EXPECTED_CMAPSS_COLUMN_COUNT
from aeroguard.data.loader import load_cmapss_dataset, read_cmapss_table
from aeroguard.data.validation import CMapssValidationError


def _row(unit_id: int, cycle: int, offset: float = 0.0) -> list[float]:
    return [
        unit_id,
        cycle,
        0.1 + offset,
        0.2 + offset,
        0.3 + offset,
        *[sensor + cycle * 0.01 + offset for sensor in range(1, 22)],
    ]


def _write_rows(path: Path, rows: list[list[float]]) -> None:
    lines = ["   ".join(str(value) for value in row) + "   " for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_cmapss_table_handles_variable_whitespace(tmp_path: Path) -> None:
    path = tmp_path / "train_FD001.txt"
    _write_rows(path, [_row(1, 1), _row(1, 2), _row(2, 1)])

    frame = read_cmapss_table(path)

    assert frame.shape == (3, EXPECTED_CMAPSS_COLUMN_COUNT)
    assert frame["unit_id"].tolist() == [1, 1, 2]
    assert frame["cycle"].tolist() == [1, 2, 1]


def test_read_cmapss_table_rejects_wrong_column_count(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("1 1 0.1\n", encoding="utf-8")

    with pytest.raises(CMapssValidationError, match="expected 26"):
        read_cmapss_table(path)


def test_read_cmapss_table_rejects_non_monotonic_cycles(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    _write_rows(path, [_row(1, 2), _row(1, 1)])

    with pytest.raises(CMapssValidationError, match="strictly increasing"):
        read_cmapss_table(path)


def test_load_cmapss_dataset_rejects_mismatched_rul_rows(tmp_path: Path) -> None:
    _write_rows(tmp_path / "train_FD001.txt", [_row(1, 1), _row(1, 2)])
    _write_rows(tmp_path / "test_FD001.txt", [_row(1, 1), _row(2, 1)])
    (tmp_path / "RUL_FD001.txt").write_text("5\n", encoding="utf-8")

    with pytest.raises(CMapssValidationError, match="RUL file has 1 rows"):
        load_cmapss_dataset(tmp_path, "FD001")
