import pandas as pd

from aeroguard.evaluation.bootstrap import bootstrap_engine_metrics


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [1, 2, 3, 4],
            "detected": [True, True, False, False],
            "lead_time_before_failure": [50.0, 30.0, None, None],
        }
    )


def test_bootstrap_is_deterministic_and_orders_intervals() -> None:
    funcs = {
        "detection_rate": lambda frame: float(frame["detected"].mean()),
        "median_lead_time": lambda frame: None
        if frame["lead_time_before_failure"].dropna().empty
        else float(frame["lead_time_before_failure"].dropna().median()),
    }

    first = bootstrap_engine_metrics(_frame(), funcs, n_samples=100, confidence_level=0.95, seed=7)
    second = bootstrap_engine_metrics(_frame(), funcs, n_samples=100, confidence_level=0.95, seed=7)

    assert first == second
    assert first["detection_rate"]["ci_lower"] <= first["detection_rate"]["estimate"] <= first["detection_rate"]["ci_upper"]
    assert first["median_lead_time"]["valid_replicates"] > 0


def test_bootstrap_handles_undefined_metric_and_empty_input() -> None:
    funcs = {"undefined": lambda frame: None}

    result = bootstrap_engine_metrics(_frame(), funcs, n_samples=10, confidence_level=0.9, seed=1)
    empty = bootstrap_engine_metrics(pd.DataFrame(columns=["unit_id"]), funcs, n_samples=10, confidence_level=0.9, seed=1)

    assert result["undefined"]["estimate"] is None
    assert result["undefined"]["valid_replicates"] == 0
    assert empty["undefined"]["valid_replicates"] == 0
