"""Row-level anomaly and engine-level onset metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from aeroguard.data.columns import CYCLE_COLUMN, UNIT_COLUMN


def row_level_anomaly_metrics(
    y_true: Any,
    y_flag: Any,
    y_score: Any | None = None,
) -> dict[str, float | int | None]:
    """Compute safe row-level binary anomaly metrics."""
    true = np.asarray(y_true, dtype=int)
    flag = np.asarray(y_flag, dtype=int)
    if len(true) == 0 or len(true) != len(flag):
        raise ValueError("Metric inputs must be non-empty and aligned.")
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(true, flag, labels=labels).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        true,
        flag,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    specificity = tn / (tn + fp) if (tn + fp) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    fnr = fn / (fn + tp) if (fn + tp) else None
    balanced_terms = []
    if recall is not None:
        balanced_terms.append(float(recall))
    if specificity is not None:
        balanced_terms.append(float(specificity))
    score_values = None if y_score is None else np.asarray(y_score, dtype=float)
    has_both_classes = len(np.unique(true)) == 2
    roc_auc = None
    pr_auc = None
    if has_both_classes and score_values is not None:
        roc_auc = float(roc_auc_score(true, score_values))
        pr_auc = float(average_precision_score(true, score_values))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": None if specificity is None else float(specificity),
        "false_positive_rate": None if fpr is None else float(fpr),
        "false_negative_rate": None if fnr is None else float(fnr),
        "balanced_accuracy": float(np.mean(balanced_terms)) if balanced_terms else None,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "support": int(len(true)),
        "positive_support": int(true.sum()),
    }


def summarize_engine_onsets(
    frame: pd.DataFrame,
    detection_flag_column: str,
    method_name: str,
    split_name: str,
    healthy_rul_threshold: float,
    critical_rul_threshold: float,
    rul_column: str = "true_rul_uncapped",
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    """Summarize engine-wise detection delay and lead time."""
    rows = []
    for unit_id, group in frame.sort_values([UNIT_COLUMN, CYCLE_COLUMN]).groupby(UNIT_COLUMN):
        first_cycle = int(group[CYCLE_COLUMN].iloc[0])
        degrading = group[group[rul_column] <= healthy_rul_threshold]
        if degrading.empty:
            proxy_onset_cycle = None
            left_censored = False
            onset_observable = False
        else:
            proxy_onset_cycle = int(degrading[CYCLE_COLUMN].iloc[0])
            left_censored = proxy_onset_cycle == first_cycle
            onset_observable = not left_censored

        detections = group[group[detection_flag_column].astype(bool)]
        if detections.empty:
            detection_cycle = None
            true_rul_at_detection = None
        else:
            first_detection = detections.iloc[0]
            detection_cycle = int(first_detection[CYCLE_COLUMN])
            true_rul_at_detection = float(first_detection[rul_column])

        if detection_cycle is not None and proxy_onset_cycle is not None and not left_censored:
            detection_delay = detection_cycle - proxy_onset_cycle
        else:
            detection_delay = None

        false_alarm = (
            detection_cycle is not None
            and proxy_onset_cycle is not None
            and not left_censored
            and detection_cycle < proxy_onset_cycle
        )
        rows.append(
            {
                "split": split_name,
                "method": method_name,
                "unit_id": int(unit_id),
                "first_alarm_cycle": detection_cycle,
                "estimated_onset_cycle": detection_cycle,
                "proxy_onset_cycle": proxy_onset_cycle,
                "left_censored": bool(left_censored),
                "onset_observable": bool(onset_observable),
                "detected": detection_cycle is not None,
                "missed": proxy_onset_cycle is not None and detection_cycle is None,
                "false_alarm": bool(false_alarm),
                "detection_delay": detection_delay,
                "true_rul_at_detection": true_rul_at_detection,
                "lead_time_before_failure": true_rul_at_detection,
                "before_critical_threshold": None
                if true_rul_at_detection is None
                else bool(true_rul_at_detection > critical_rul_threshold),
                "before_60_cycles_rul": None
                if true_rul_at_detection is None
                else bool(true_rul_at_detection > 60),
                "before_30_cycles_rul": None
                if true_rul_at_detection is None
                else bool(true_rul_at_detection > 30),
            }
        )
    summary = pd.DataFrame(rows)
    observed = summary[summary["onset_observable"]]
    detected = summary[summary["detected"]]
    delays = observed["detection_delay"].dropna()
    lead_times = detected["lead_time_before_failure"].dropna()

    def true_count(column: str) -> int:
        return int(sum(value is True for value in detected[column].tolist()))

    metrics = {
        "engines_evaluated": int(summary["unit_id"].nunique()),
        "engines_with_proxy_degradation_onset_observable": int(observed["unit_id"].nunique()),
        "left_censored_engines": int(summary["left_censored"].sum()),
        "detected_engines": int(summary["detected"].sum()),
        "missed_engines": int(summary["missed"].sum()),
        "detection_rate": None
        if len(summary) == 0
        else float(summary["detected"].mean()),
        "false_alarm_engine_count": int(summary["false_alarm"].sum()),
        "median_detection_delay": None if delays.empty else float(delays.median()),
        "mean_detection_delay": None if delays.empty else float(delays.mean()),
        "median_lead_time": None if lead_times.empty else float(lead_times.median()),
        "mean_lead_time": None if lead_times.empty else float(lead_times.mean()),
        "detections_before_critical_threshold": true_count("before_critical_threshold"),
        "detections_before_60_cycles_rul": true_count("before_60_cycles_rul"),
        "detections_before_30_cycles_rul": true_count("before_30_cycles_rul"),
    }
    return summary, metrics
