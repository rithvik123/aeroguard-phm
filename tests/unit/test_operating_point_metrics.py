from aeroguard.evaluation.operating_point_metrics import compute_operating_utility


def test_operating_utility_rewards_detection_and_penalizes_false_alarm() -> None:
    weights = {
        "detection_reward": 1.0,
        "early_warning_reward": 0.5,
        "missed_engine_penalty": 1.0,
        "false_alarm_engine_penalty": 1.0,
        "healthy_fpr_penalty": 1.0,
        "instability_penalty": 0.1,
        "late_after_critical_penalty": 0.2,
    }
    good = compute_operating_utility(
        {"false_positive_rate": 0.01},
        {
            "engines_evaluated": 10,
            "detected_engines": 9,
            "missed_engines": 1,
            "false_alarm_engine_count": 0,
            "detections_before_30_cycles_rul": 8,
            "detection_rate": 0.9,
        },
        transition_count=1,
        weights=weights,
    )
    poor = compute_operating_utility(
        {"false_positive_rate": 0.2},
        {
            "engines_evaluated": 10,
            "detected_engines": 3,
            "missed_engines": 7,
            "false_alarm_engine_count": 4,
            "detections_before_30_cycles_rul": 1,
            "detection_rate": 0.3,
        },
        transition_count=20,
        weights=weights,
    )

    assert good > poor
