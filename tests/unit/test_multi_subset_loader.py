from pathlib import Path

import pytest

from aeroguard.data.multi_subset import (
    add_subset_identity,
    load_test_subsets,
    load_training_subsets,
    validate_global_engine_index,
)


def _row(unit_id: int, cycle: int, offset: float = 0.0) -> list[float]:
    degradation = max(cycle - 3, 0)
    return [
        unit_id,
        cycle,
        0.1 * unit_id + offset,
        0.2 * cycle + offset,
        1.0 + offset,
        *[
            float(sensor + 0.03 * unit_id + 0.02 * cycle + 0.01 * degradation * sensor + offset)
            for sensor in range(1, 22)
        ],
    ]


def _write_rows(path: Path, rows: list[list[float]]) -> None:
    path.write_text("\n".join(" ".join(str(value) for value in row) for row in rows) + "\n", encoding="utf-8")


def _write_subset(root: Path, subset: str, offset: float = 0.0) -> None:
    train_rows = [_row(unit, cycle, offset) for unit in [1, 2] for cycle in range(1, 6)]
    test_rows = [_row(unit, cycle, offset) for unit in [1, 2] for cycle in range(1, 4)]
    _write_rows(root / f"train_{subset}.txt", train_rows)
    _write_rows(root / f"test_{subset}.txt", test_rows)
    (root / f"RUL_{subset}.txt").write_text("2\n3\n", encoding="utf-8")


def test_load_training_subsets_adds_collision_safe_global_ids(tmp_path: Path) -> None:
    _write_subset(tmp_path, "FD001", 0.0)
    _write_subset(tmp_path, "FD002", 1.0)

    frame, metadata = load_training_subsets(tmp_path, ["FD001", "FD002"], 4, 4, 2)

    assert frame["source_domain"].nunique() == 2
    assert {"FD001_0001", "FD001_0002", "FD002_0001", "FD002_0002"} == set(frame["global_engine_id"])
    assert frame[["global_engine_id", "cycle"]].duplicated().sum() == 0
    assert metadata["combined_engine_count"] == 4
    assert {"proxy_degradation_label", "proxy_critical_label", "local_unit_id"}.issubset(frame.columns)


def test_load_test_subsets_derives_final_rul_and_identity(tmp_path: Path) -> None:
    _write_subset(tmp_path, "FD001", 0.0)
    _write_subset(tmp_path, "FD003", 2.0)

    frames, metadata = load_test_subsets(tmp_path, ["FD001", "FD003"], 4, 2)

    fd003 = frames["FD003"].sort_values(["global_engine_id", "cycle"])
    final_rul = fd003.groupby("global_engine_id").tail(1)["true_rul_uncapped"].tolist()
    assert final_rul == [2.0, 3.0]
    assert metadata["FD003"]["test_engine_count"] == 2
    assert fd003["subset"].eq("FD003").all()


def test_validate_global_engine_index_rejects_duplicate_engine_cycle() -> None:
    frame = add_subset_identity(
        __import__("pandas").DataFrame({"unit_id": [1, 1], "cycle": [1, 1]}),
        "FD001",
    )

    with pytest.raises(ValueError, match="Duplicate global-engine-cycle"):
        validate_global_engine_index(frame)
