import pandas as pd
import pytest

from aeroguard.anomaly.policy_registry import apply_alert_policy, validate_policy_registry


PROFILE = {
    "balanced": {
        "detection_rate": 1.0,
        "critical_region_recall": 1.0,
        "missed_engine_rate": 1.0,
        "false_alarm_engine_rate": 1.0,
        "healthy_region_false_positive_rate": 1.0,
        "detected_before_30_fraction": 1.0,
        "detected_before_60_fraction": 1.0,
        "alert_instability": 1.0,
        "utility_variability": 1.0,
    }
}


def _policy(**overrides):
    policy = {
        "policy_id": "demo",
        "calibration_method": "empirical_percentile",
        "fusion_method": "weighted_mean",
        "weights": {"pca_reconstruction": 0.0, "isolation_forest": 0.5, "one_class_svm": 0.5},
        "threshold": 0.8,
        "persistence": {"type": "consecutive", "k": 2},
        "hysteresis": {
            "enter_threshold": 0.8,
            "exit_threshold": 0.6,
            "min_enter_duration": 1,
            "min_clear_duration": 1,
        },
        "operational_profile": "balanced",
    }
    policy.update(overrides)
    return policy


def test_valid_registry_and_policy_application() -> None:
    resolved = validate_policy_registry([_policy()], 3, PROFILE)
    frame = pd.DataFrame(
        {
            "unit_id": [1, 1, 1],
            "cycle": [1, 2, 3],
            "pca_reconstruction_calibrated_score": [0.1, 0.1, 0.1],
            "isolation_forest_calibrated_score": [0.9, 0.9, 0.2],
            "one_class_svm_calibrated_score": [0.9, 0.9, 0.2],
            "smoothed_health_index": [0.7, 0.5, 0.5],
        }
    )

    result, summary = apply_alert_policy(
        frame,
        resolved[0],
        {
            "monitor_score": 0.6,
            "warning_score": 0.8,
            "critical_score": 0.95,
            "warning_health_index_max": 0.55,
            "critical_health_index_max": 0.35,
        },
    )

    assert result["locked_persistent_alarm_state"].tolist() == [False, True, False]
    assert summary["number_of_alarm_transitions"].iloc[0] == 2


def test_duplicate_policy_ids_raise() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        validate_policy_registry([_policy(), _policy()], 3, PROFILE)


def test_invalid_weights_raise() -> None:
    bad = _policy(weights={"pca_reconstruction": 0.5, "isolation_forest": 0.5, "one_class_svm": 0.5})
    with pytest.raises(ValueError, match="sum"):
        validate_policy_registry([bad], 3, PROFILE)


def test_invalid_threshold_persistence_hysteresis_and_count_raise() -> None:
    with pytest.raises(ValueError, match="threshold"):
        validate_policy_registry([_policy(threshold=1.2)], 3, PROFILE)
    with pytest.raises(ValueError, match="persistence"):
        validate_policy_registry([_policy(persistence={"type": "k_of_n", "k": 4, "n": 3})], 3, PROFILE)
    with pytest.raises(ValueError, match="Hysteresis"):
        validate_policy_registry([_policy(hysteresis={"enter_threshold": 0.5, "exit_threshold": 0.6, "min_enter_duration": 1, "min_clear_duration": 1})], 3, PROFILE)
    with pytest.raises(ValueError, match="maximum"):
        validate_policy_registry([_policy()], 0, PROFILE)
