import math

import pandas as pd

from aeroguard.evaluation.anomaly_metrics import (
    row_level_anomaly_metrics,
    summarize_engine_onsets,
)


def test_row_level_metrics_with_both_classes() -> None:
    metrics = row_level_anomaly_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        [0.1, 0.8, 0.9, 0.2],
    )

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["roc_auc"] is not None
    assert metrics["pr_auc"] is not None


def test_row_level_metrics_single_class_has_no_auc() -> None:
    metrics = row_level_anomaly_metrics([0, 0, 0], [0, 1, 0], [0.1, 0.9, 0.2])

    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None


def test_engine_onset_summary_delay_and_false_alarm() -> None:
    frame = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3],
            "true_rul_uncapped": [130, 120, 20, 100, 90, 20],
            "alarm": [1, 0, 0, 0, 0, 1],
        }
    )

    summary, metrics = summarize_engine_onsets(
        frame,
        "alarm",
        "demo",
        "validation",
        healthy_rul_threshold=125,
        critical_rul_threshold=30,
    )

    engine_1 = summary[summary["unit_id"] == 1].iloc[0]
    assert engine_1["detection_delay"] == -1
    assert engine_1["false_alarm"]
    assert metrics["engines_evaluated"] == 2
    assert metrics["left_censored_engines"] == 1
    assert math.isclose(metrics["detection_rate"], 1.0)
