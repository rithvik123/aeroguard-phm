import pandas as pd
import pytest

from aeroguard.data.targets import add_training_rul_targets, final_observed_test_rows
from aeroguard.data.validation import CMapssValidationError


def _frame(rows: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [unit for unit, _ in rows],
            "cycle": [cycle for _, cycle in rows],
            "operational_setting_1": 0.0,
            "operational_setting_2": 0.0,
            "operational_setting_3": 0.0,
            "sensor_1": 1.0,
        }
    )


def test_training_rul_for_one_engine_and_final_cycle_zero() -> None:
    result = add_training_rul_targets(_frame([(1, 1), (1, 2), (1, 3)]), rul_cap=125)

    assert result["rul_uncapped"].tolist() == [2, 1, 0]
    assert result["rul_capped"].tolist() == [2, 1, 0]
    assert result.loc[result["cycle"] == 3, "rul_uncapped"].iloc[0] == 0


def test_training_rul_multiple_engines_with_different_lengths() -> None:
    result = add_training_rul_targets(
        _frame([(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (2, 4)]),
        rul_cap=125,
    )

    assert result["rul_uncapped"].tolist() == [1, 0, 3, 2, 1, 0]


def test_training_rul_clipping_preserves_uncapped_target() -> None:
    result = add_training_rul_targets(_frame([(1, 1), (1, 2), (1, 3), (1, 4)]), rul_cap=2)

    assert result["rul_uncapped"].tolist() == [3, 2, 1, 0]
    assert result["rul_capped"].tolist() == [2, 2, 1, 0]


def test_test_final_rul_mapping_uses_engine_order() -> None:
    test = _frame([(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)])
    result = final_observed_test_rows(test, pd.Series([7, 11]))

    assert result["unit_id"].tolist() == [1, 2]
    assert result["cycle"].tolist() == [2, 3]
    assert result["test_final_rul"].tolist() == [7.0, 11.0]


def test_test_final_rul_rejects_mismatched_count() -> None:
    test = _frame([(1, 1), (2, 1)])

    with pytest.raises(CMapssValidationError, match="RUL file has 1 rows"):
        final_observed_test_rows(test, pd.Series([7]))


def test_test_final_rul_rejects_negative_values() -> None:
    test = _frame([(1, 1)])

    with pytest.raises(CMapssValidationError, match="non-negative"):
        final_observed_test_rows(test, pd.Series([-1]))
