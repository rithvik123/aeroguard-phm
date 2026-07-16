from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from aeroguard.pipelines.refine_maintenance_safety_policy import (
    _synthetic_frames,
    apply_policy,
    assign_safety_state,
    candidate_feasibility,
    load_config,
    materialize_policy_for_json,
    policy_metrics,
    prepare_decision_frame,
    run_smoke_test,
    split_engines,
    stable_hash,
    wilson_interval,
)


def _config() -> dict:
    return load_config("configs/maintenance_safety_policy_refinement.yaml")


def _decision_frame() -> pd.DataFrame:
    config = _config()
    cv, _, uncertainty, abstention = _synthetic_frames()
    support = {"low_support_rarity_threshold": 0.99, "high_width_threshold": 999.0, "low_lower_bound_threshold": 20.0}
    return prepare_decision_frame(cv, uncertainty, abstention, support, config)


def test_safety_state_assignment_and_threshold_sensitivity() -> None:
    config = _config()
    states = assign_safety_state(pd.Series([5, 15, 16, 30, 31, 60, 61, 90, 91]), config)
    assert states.tolist() == [
        "CRITICAL",
        "CRITICAL",
        "NEAR_TERM",
        "NEAR_TERM",
        "INSPECTION_WINDOW",
        "INSPECTION_WINDOW",
        "MONITORING",
        "MONITORING",
        "HEALTHY",
    ]
    assert config["safety_states"]["critical_threshold_sensitivity"] == [10, 15, 20]


def test_engine_grouping_has_no_overlap() -> None:
    frame = _decision_frame()
    dev, val, split = split_engines(frame, 0.7, 123)
    assert split["engine_overlap_count"] == 0
    assert set(dev["global_engine_id"]).isdisjoint(set(val["global_engine_id"])) or set(dev["subset"] + dev["global_engine_id"]).isdisjoint(set(val["subset"] + val["global_engine_id"]))


def test_point_lower_and_risk_policy_behaviour() -> None:
    frame = _decision_frame().head(4).copy()
    frame["predicted_rul"] = [8.0, 25.0, 50.0, 100.0]
    frame["lower_90"] = [6.0, 10.0, 35.0, 80.0]
    frame["high_error_risk_probability"] = [0.9, 0.9, 0.1, 0.1]
    frame["abstain_flag"] = False
    frame["low_support_condition"] = False
    frame["unsupported_operating_condition"] = False
    frame["invalid_uncertainty"] = False
    point = apply_policy(frame, {"policy_id": "p", "family": "point_threshold", "thresholds": {"tc": 10, "tm": 30, "ti": 60}})
    assert point["maintenance_action"].tolist() == ["URGENT_ENGINEERING_REVIEW", "SCHEDULE_MAINTENANCE", "PLAN_INSPECTION", "CONTINUE_MONITORING"]
    lower = apply_policy(frame, {"policy_id": "l", "family": "lower_bound_threshold", "thresholds": {"lc": 10, "lm": 30, "li": 60}})
    assert lower["maintenance_action"].iloc[1] == "URGENT_ENGINEERING_REVIEW"
    risk = apply_policy(frame, {"policy_id": "r", "family": "risk_aware_hybrid", "thresholds": {"tc": 10, "lc": 15, "risk": 0.8, "tm": 30, "li": 60}})
    assert risk["maintenance_action"].iloc[1] == "URGENT_ENGINEERING_REVIEW"


def test_abstained_and_low_support_predictions_receive_mandatory_review() -> None:
    frame = _decision_frame().head(2).copy()
    frame["predicted_rul"] = 100.0
    frame["lower_90"] = 90.0
    frame["abstain_flag"] = [True, False]
    frame["low_support_condition"] = [False, True]
    frame["unsupported_operating_condition"] = False
    frame["invalid_uncertainty"] = False
    result = apply_policy(frame, {"policy_id": "p", "family": "point_threshold", "thresholds": {"tc": 10, "tm": 30, "ti": 60}})
    assert result["maintenance_action"].tolist() == ["ABSTAIN_AND_REVIEW", "ABSTAIN_AND_REVIEW"]


def test_operational_and_direct_recall_are_distinct() -> None:
    config = _config()
    frame = _decision_frame().head(3).copy()
    frame["safety_state"] = ["CRITICAL", "CRITICAL", "HEALTHY"]
    frame["maintenance_action"] = ["URGENT_ENGINEERING_REVIEW", "ABSTAIN_AND_REVIEW", "CONTINUE_MONITORING"]
    metrics = policy_metrics(frame, config)
    assert metrics["direct_urgent_critical_recall"] == 0.5
    assert metrics["operational_critical_recall"] == 1.0
    assert metrics["critical_captured_by_abstain_review"] == 1


def test_monotone_policy_directions_do_not_reduce_urgency() -> None:
    frame = _decision_frame().head(2).copy()
    frame["predicted_rul"] = [100.0, 5.0]
    frame["lower_90"] = [90.0, 0.0]
    frame["high_error_risk_probability"] = [0.0, 1.0]
    frame["support_category_code"] = [0.0, 3.0]
    frame["degradation_rate_filled"] = [0.0, 1.0]
    frame[["abstain_flag", "low_support_condition", "unsupported_operating_condition", "invalid_uncertainty"]] = False
    result = apply_policy(frame, {"policy_id": "m", "family": "monotone_score", "weights": {"rul": 0.4, "lower_bound": 0.3, "risk": 0.2, "support": 0.1}, "thresholds": {"urgent": 0.65, "schedule": 0.45, "plan": 0.25}})
    ranks = result["maintenance_action"].map({"CONTINUE_MONITORING": 0, "PLAN_INSPECTION": 1, "SCHEDULE_MAINTENANCE": 2, "URGENT_ENGINEERING_REVIEW": 3})
    assert ranks.iloc[1] >= ranks.iloc[0]


def test_safety_floors_are_enforced_without_weakening() -> None:
    config = _config()
    feasible, reasons = candidate_feasibility({"operational_critical_recall": 0.5, "missed_critical_rate": 0.5, "urgent_review_precision": 0.1, "urgent_review_rate": 0.1, "total_review_workload": 0.1}, config)
    assert feasible is False
    assert "minimum_critical_recall" in reasons
    assert "maximum_missed_critical_rate" in reasons


def test_wilson_interval_and_policy_hash_are_deterministic() -> None:
    assert wilson_interval(9, 10)[0] < 0.9 < wilson_interval(9, 10)[1]
    payload = {"policy_id": "x", "thresholds": {"a": 1.0}, "lock_timestamp": "later"}
    assert stable_hash(payload) == stable_hash(dict(payload, lock_timestamp="different"))
    locked = materialize_policy_for_json({"policy_id": "x", "policy_family": "point", "thresholds": {"a": 1.0}})
    assert locked["policy_hash"] == stable_hash(locked)


def test_smoke_completes_without_neural_training(tmp_path: Path) -> None:
    config = _config()
    config["outputs"]["reports_dir"] = str(tmp_path / "reports")
    config["outputs"]["artifacts_dir"] = str(tmp_path / "artifacts")
    path = tmp_path / "safety.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = run_smoke_test(path)
    assert result["status"] == "smoke_complete"
    assert result["policy_locked_before_benchmark"] is True
    assert result["neural_training_function_called"] is False
