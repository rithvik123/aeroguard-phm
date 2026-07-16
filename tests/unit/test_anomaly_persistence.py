import pandas as pd

from aeroguard.anomaly.persistence import apply_persistent_alarms


def _frame(flags: list[bool], cycles: list[int] | None = None, unit_id: int = 1) -> pd.DataFrame:
    if cycles is None:
        cycles = list(range(1, len(flags) + 1))
    return pd.DataFrame({"unit_id": unit_id, "cycle": cycles, "flag": flags})


def test_no_anomalies_create_no_alarm() -> None:
    result, summary = apply_persistent_alarms(_frame([False, False, False]), "flag", "demo", 2)

    assert not result["demo_persistent_alarm_flag"].any()
    assert summary["persistent_alarm_detected"].tolist() == [False]


def test_isolated_anomaly_does_not_persist() -> None:
    result, _ = apply_persistent_alarms(_frame([False, True, False, True]), "flag", "demo", 2)

    assert not result["demo_persistent_alarm_flag"].any()


def test_exact_required_run_marks_start_cycle() -> None:
    result, summary = apply_persistent_alarms(_frame([False, True, True]), "flag", "demo", 2)

    assert summary["first_persistent_alarm_cycle"].iloc[0] == 2
    assert result.loc[result["cycle"] == 2, "demo_persistent_alarm_started"].iloc[0]
    assert result.loc[result["cycle"] >= 2, "demo_persistent_alarm_flag"].all()


def test_longer_run_and_multiple_runs_use_first_run() -> None:
    result, summary = apply_persistent_alarms(
        _frame([True, True, True, False, True, True]),
        "flag",
        "demo",
        2,
    )

    assert summary["first_persistent_alarm_cycle"].iloc[0] == 1
    assert result["demo_persistent_alarm_started"].sum() == 1


def test_independent_engines_and_unsorted_rows() -> None:
    frame = pd.concat(
        [
            _frame([True, True], [2, 1], unit_id=1),
            _frame([False, True, True], [1, 3, 2], unit_id=2),
        ],
        ignore_index=True,
    )
    result, summary = apply_persistent_alarms(frame, "flag", "demo", 2)

    assert set(summary["unit_id"]) == {1, 2}
    assert result.groupby("unit_id")["demo_persistent_alarm_started"].sum().to_dict() == {1: 1, 2: 1}


def test_cycle_gaps_break_consecutive_runs_by_default() -> None:
    result, _ = apply_persistent_alarms(_frame([True, True], cycles=[1, 3]), "flag", "demo", 2)

    assert not result["demo_persistent_alarm_flag"].any()
