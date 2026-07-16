import pandas as pd
import pytest

from aeroguard.maintenance.uncertainty_policy import assign_maintenance_recommendations, maintenance_policy_metrics


THRESHOLDS = {"urgent_review_max": 15, "schedule_maintenance_max": 30, "plan_inspection_max": 60}


def test_maintenance_policy_thresholds_and_abstention_override() -> None:
    frame = pd.DataFrame(
        {
            "lower_90": [70.0, 45.0, 20.0, 10.0, 80.0],
            "abstain_flag": [False, False, False, False, True],
            "true_rul": [80.0, 50.0, 25.0, 10.0, 90.0],
            "predicted_rul": [75.0, 50.0, 20.0, 10.0, 85.0],
            "subset": ["FD001"] * 5,
        }
    )

    result = assign_maintenance_recommendations(frame, THRESHOLDS)

    assert result["maintenance_action"].tolist() == [
        "CONTINUE_MONITORING",
        "PLAN_INSPECTION",
        "SCHEDULE_MAINTENANCE",
        "URGENT_ENGINEERING_REVIEW",
        "ENGINEERING_REVIEW_REQUIRED",
    ]
    assert "not approved" in result["maintenance_disclaimer"].iloc[0]


def test_maintenance_metrics_and_invalid_threshold_ordering() -> None:
    frame = assign_maintenance_recommendations(
        pd.DataFrame(
            {
                "lower_90": [10.0],
                "abstain_flag": [False],
                "true_rul": [12.0],
                "predicted_rul": [14.0],
                "subset": ["FD001"],
            }
        ),
        THRESHOLDS,
    )
    metrics = maintenance_policy_metrics(frame)

    assert metrics["urgent_review_recall_true_rul_le_15"] == 1.0
    with pytest.raises(ValueError):
        assign_maintenance_recommendations(frame, {"urgent_review_max": 30, "schedule_maintenance_max": 20, "plan_inspection_max": 60})
