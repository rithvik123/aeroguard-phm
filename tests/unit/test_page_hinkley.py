import pandas as pd

from aeroguard.onset.onset_detection import apply_page_hinkley_by_engine
from aeroguard.onset.page_hinkley import PageHinkley


def test_stable_signal_has_no_change() -> None:
    detector = PageHinkley(delta=0.0, threshold=1.0, min_observations=5)

    assert not any(detector.run([1.0] * 30))


def test_clear_upward_shift_is_detected() -> None:
    detector = PageHinkley(delta=0.0, threshold=0.5, min_observations=5, direction="increase")
    flags = detector.run([0.0] * 10 + [2.0] * 10)

    assert any(flags[10:])


def test_clear_downward_shift_is_detected() -> None:
    detector = PageHinkley(delta=0.0, threshold=0.5, min_observations=5, direction="decrease")
    flags = detector.run([2.0] * 10 + [0.0] * 10)

    assert any(flags[10:])


def test_minimum_observation_behavior() -> None:
    detector = PageHinkley(delta=0.0, threshold=0.1, min_observations=8, direction="increase")
    flags = detector.run([0.0] * 3 + [10.0] * 3)

    assert not any(flags)


def test_engine_wise_state_is_independent_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "unit_id": [1] * 20 + [2] * 20,
            "cycle": list(range(1, 21)) * 2,
            "signal": [0.0] * 10 + [2.0] * 10 + [1.0] * 20,
        }
    )

    result_a, summary_a = apply_page_hinkley_by_engine(
        frame,
        "signal",
        "ph",
        delta=0.0,
        threshold=0.5,
        min_observations=5,
        direction="increase",
    )
    result_b, summary_b = apply_page_hinkley_by_engine(
        frame,
        "signal",
        "ph",
        delta=0.0,
        threshold=0.5,
        min_observations=5,
        direction="increase",
    )

    assert result_a["ph_change_flag"].tolist() == result_b["ph_change_flag"].tolist()
    assert summary_a.loc[summary_a["unit_id"] == 1, "detected"].iloc[0]
    assert not summary_a.loc[summary_a["unit_id"] == 2, "detected"].iloc[0]
    assert summary_a.equals(summary_b)
