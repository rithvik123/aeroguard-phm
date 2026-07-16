import pandas as pd
import pytest

from aeroguard.anomaly.alerting import apply_hysteresis_alert, apply_persistence_rule


def _frame(scores: list[float], unit_id: int = 1, cycles: list[int] | None = None) -> pd.DataFrame:
    if cycles is None:
        cycles = list(range(1, len(scores) + 1))
    return pd.DataFrame(
        {
            "unit_id": unit_id,
            "cycle": cycles,
            "score": scores,
            "flag": [score >= 0.8 for score in scores],
        }
    )


def test_hysteresis_stable_healthy_sequence() -> None:
    result, summary = apply_hysteresis_alert(_frame([0.1, 0.2, 0.3]), "score", "demo", 0.8, 0.5, 2, 2)

    assert not result["demo_alert_state"].any()
    assert not summary["alert_ever_active"].iloc[0]


def test_hysteresis_entry_temporary_dip_clear_and_reentry() -> None:
    result, summary = apply_hysteresis_alert(
        _frame([0.9, 0.91, 0.7, 0.86, 0.4, 0.3, 0.9, 0.91]),
        "score",
        "demo",
        0.85,
        0.5,
        2,
        2,
    )

    assert result["demo_alert_started"].sum() == 2
    assert result["demo_alert_cleared"].sum() == 1
    assert summary["number_of_transitions"].iloc[0] == 3


def test_hysteresis_independent_engine_state() -> None:
    frame = pd.concat([_frame([0.9, 0.91], 1), _frame([0.1, 0.2], 2)], ignore_index=True)
    result, summary = apply_hysteresis_alert(frame, "score", "demo", 0.85, 0.5, 2, 2)

    assert result.groupby("unit_id")["demo_alert_state"].any().to_dict() == {1: True, 2: False}
    assert summary.set_index("unit_id")["alert_ever_active"].to_dict() == {1: True, 2: False}


def test_invalid_hysteresis_threshold_ordering() -> None:
    with pytest.raises(ValueError, match="exit_threshold"):
        apply_hysteresis_alert(_frame([0.1]), "score", "demo", 0.5, 0.6, 1, 1)


def test_consecutive_and_k_of_n_persistence_with_cycle_gaps() -> None:
    frame = _frame([0.9, 0.9, 0.1, 0.9, 0.9], cycles=[1, 2, 3, 5, 6])
    consecutive, _ = apply_persistence_rule(
        frame,
        "flag",
        "score",
        "consec",
        {"name": "consecutive_2", "type": "consecutive", "k": 2},
    )
    kofn, _ = apply_persistence_rule(
        frame,
        "flag",
        "score",
        "kofn",
        {"name": "2_of_3", "type": "k_of_n", "k": 2, "n": 3},
    )

    assert consecutive["consec_persistent_alarm_started"].sum() == 2
    assert not consecutive.loc[consecutive["cycle"] == 5, "consec_persistent_alarm_started"].any()
    assert not kofn.loc[kofn["cycle"] == 5, "kofn_persistent_alarm_started"].any()


def test_persistence_records_clear_and_reentry() -> None:
    result, summary = apply_persistence_rule(
        _frame([0.9, 0.9, 0.1, 0.9, 0.9]),
        "flag",
        "score",
        "demo",
        {"name": "consecutive_2", "type": "consecutive", "k": 2},
    )

    assert result["demo_persistent_alarm_started"].sum() == 2
    assert result["demo_persistent_alarm_cleared"].sum() == 1
    assert summary["number_of_alarm_transitions"].iloc[0] == 3
    assert summary["alarm_cleared"].iloc[0]
    assert summary["alarm_reappeared"].iloc[0]


def test_alarm_at_end_of_trajectory() -> None:
    result, summary = apply_persistence_rule(
        _frame([0.1, 0.9, 0.9]),
        "flag",
        "score",
        "demo",
        {"name": "consecutive_2", "type": "consecutive", "k": 2},
    )

    assert result["demo_persistent_alarm_started"].sum() == 1
    assert summary["first_persistent_alarm_cycle"].iloc[0] == 2
