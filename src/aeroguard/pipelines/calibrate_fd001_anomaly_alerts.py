"""Validation-calibrated anomaly fusion and operational alerting for FD001."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from aeroguard.anomaly.alerting import (
    apply_hysteresis_alert,
    apply_persistence_rule,
    assign_operational_alert_levels,
)
from aeroguard.anomaly.ensemble import DETECTOR_ORDER, fuse_scores, validate_weights, voting_flags, voting_score
from aeroguard.anomaly.score_calibration import ScoreCalibrator
from aeroguard.data.columns import CYCLE_COLUMN, UNIT_COLUMN
from aeroguard.evaluation.alert_metrics import alert_transition_metrics
from aeroguard.evaluation.anomaly_metrics import row_level_anomaly_metrics, summarize_engine_onsets
from aeroguard.evaluation.operating_point_metrics import compute_operating_utility
from aeroguard.onset.onset_detection import apply_page_hinkley_by_engine
from aeroguard.pipelines.train_fd001_baseline import project_root, resolve_project_path
from aeroguard.pipelines.train_fd001_health_anomaly import (
    apply_health_and_detectors,
    load_config as load_phase2_config,
    prepare_datasets,
    select_anomaly_features,
)


RAW_SCORE_COLUMNS = {
    "pca_reconstruction": "pca_anomaly_score",
    "isolation_forest": "isolation_forest_score",
    "one_class_svm": "one_class_svm_score",
}

NATIVE_FLAG_COLUMNS = {
    "pca_reconstruction": "pca_anomaly_flag",
    "isolation_forest": "isolation_forest_flag",
    "one_class_svm": "one_class_svm_flag",
}

CALIBRATED_SCORE_COLUMNS = {
    name: f"{name}_calibrated_score" for name in DETECTOR_ORDER
}


REQUIRED_CONFIG_KEYS = {
    "dataset_dir",
    "subset",
    "baseline_config_path",
    "phase2_config_path",
    "random_seed",
    "validation_fraction",
    "healthy_rul_threshold",
    "critical_rul_threshold",
    "calibration_method",
    "calibration_quantiles",
    "epsilon",
    "candidate_detector_thresholds",
    "candidate_fusion_methods",
    "candidate_fusion_weights",
    "candidate_voting_rules",
    "voting_detector_threshold",
    "default_persistence",
    "persistence_base_threshold",
    "candidate_consecutive_persistence_values",
    "candidate_k_of_n_rules",
    "score_duration_rules",
    "hysteresis",
    "selected_hysteresis",
    "page_hinkley_candidates",
    "utility_profiles",
    "primary_utility_profile",
    "operational_alert_thresholds",
    "use_rul_predictions_in_alerts",
    "output_dir",
    "representative_timeline_count",
    "plotting",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(payload), handle, indent=2, allow_nan=False)


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_config(config, project_root())
    return config


def validate_config(config: dict[str, Any], root: Path) -> None:
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"Missing required configuration keys: {missing}")
    if str(config["subset"]).upper() != "FD001":
        raise ValueError("Only FD001 is supported in Phase 2B.")
    validation_fraction = float(config["validation_fraction"])
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if float(config["healthy_rul_threshold"]) <= float(config["critical_rul_threshold"]):
        raise ValueError("healthy_rul_threshold must be greater than critical_rul_threshold.")
    for key in ["dataset_dir", "baseline_config_path", "phase2_config_path"]:
        path = resolve_project_path(config[key], root)
        if not path.exists():
            raise FileNotFoundError(f"Required path not found for {key}: {path}")
    output_dir = resolve_project_path(config["output_dir"], root)
    lowered = str(output_dir).lower()
    if "\\references\\" in lowered or "\\extracted-code\\" in lowered:
        raise ValueError("Output path must not be inside read-only reference directories.")
    if config["calibration_method"] not in {"empirical_percentile", "robust_z", "quantile"}:
        raise ValueError("Invalid calibration_method.")
    quantiles = config["calibration_quantiles"]
    if not 0 <= float(quantiles["lower"]) < float(quantiles["upper"]) <= 1:
        raise ValueError("Invalid calibration quantiles.")
    if float(config["epsilon"]) <= 0:
        raise ValueError("epsilon must be positive.")
    for threshold in config["candidate_detector_thresholds"]:
        if not 0 <= float(threshold) <= 1:
            raise ValueError("Candidate detector thresholds must be in [0, 1].")
    if not config["candidate_detector_thresholds"]:
        raise ValueError("At least one candidate detector threshold is required.")
    for method in config["candidate_fusion_methods"]:
        if method not in {"mean", "median", "max", "weighted_mean", "rank_average"}:
            raise ValueError(f"Invalid fusion method: {method}")
    for weights in config["candidate_fusion_weights"]:
        validate_weights({name: weights[name] for name in DETECTOR_ORDER})
    for rule in config["candidate_voting_rules"]:
        if rule not in {"any_one", "at_least_two", "all_three"}:
            raise ValueError(f"Invalid voting rule: {rule}")
    if not 0 <= float(config["voting_detector_threshold"]) <= 1:
        raise ValueError("voting_detector_threshold must be in [0, 1].")
    if int(config["default_persistence"]["consecutive"]) <= 0:
        raise ValueError("default_persistence.consecutive must be positive.")
    if not 0 <= float(config["persistence_base_threshold"]) <= 1:
        raise ValueError("persistence_base_threshold must be in [0, 1].")
    for value in config["candidate_consecutive_persistence_values"]:
        if int(value) <= 0:
            raise ValueError("Consecutive persistence values must be positive.")
    for rule in config["candidate_k_of_n_rules"]:
        if int(rule["k"]) <= 0 or int(rule["n"]) <= 0 or int(rule["k"]) > int(rule["n"]):
            raise ValueError("Invalid K-of-N persistence rule.")
    for rule in config["score_duration_rules"]:
        if int(rule["duration"]) <= 0 or not 0 <= float(rule["threshold"]) <= 1:
            raise ValueError("Invalid score-duration persistence rule.")
    enter_thresholds = config["hysteresis"]["enter_thresholds"]
    exit_thresholds = config["hysteresis"]["exit_thresholds"]
    if not enter_thresholds or not exit_thresholds or len(enter_thresholds) != len(exit_thresholds):
        raise ValueError("Hysteresis enter and exit threshold lists must be non-empty and equal length.")
    if int(config["hysteresis"]["min_enter_duration"]) <= 0 or int(config["hysteresis"]["min_clear_duration"]) <= 0:
        raise ValueError("Hysteresis minimum durations must be positive.")
    for enter, exit_ in zip(enter_thresholds, exit_thresholds):
        if not 0 <= float(exit_) < float(enter) <= 1:
            raise ValueError("Hysteresis exit thresholds must be below enter thresholds.")
    selected_hysteresis = config["selected_hysteresis"]
    if not 0 <= float(selected_hysteresis["exit_threshold"]) < float(selected_hysteresis["enter_threshold"]) <= 1:
        raise ValueError("selected_hysteresis exit_threshold must be below enter_threshold.")
    if int(selected_hysteresis["min_enter_duration"]) <= 0 or int(selected_hysteresis["min_clear_duration"]) <= 0:
        raise ValueError("selected_hysteresis minimum durations must be positive.")
    if config["primary_utility_profile"] not in config["utility_profiles"]:
        raise ValueError("primary_utility_profile must exist in utility_profiles.")
    required_utility_weights = {
        "detection_reward",
        "early_warning_reward",
        "missed_engine_penalty",
        "false_alarm_engine_penalty",
        "healthy_fpr_penalty",
        "instability_penalty",
        "late_after_critical_penalty",
    }
    for profile, weights in config["utility_profiles"].items():
        missing_weights = sorted(required_utility_weights - set(weights))
        if missing_weights:
            raise ValueError(f"Utility profile {profile} is missing weights: {missing_weights}")
        for name, value in weights.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"Utility weight {profile}.{name} must be finite and non-negative.")
    alert_thresholds = config["operational_alert_thresholds"]
    if not 0 <= float(alert_thresholds["monitor_score"]) <= float(alert_thresholds["warning_score"]) <= float(alert_thresholds["critical_score"]) <= 1:
        raise ValueError("Operational score thresholds must satisfy monitor <= warning <= critical within [0, 1].")
    if not 0 <= float(alert_thresholds["critical_health_index_max"]) <= float(alert_thresholds["warning_health_index_max"]) <= 1:
        raise ValueError("Operational health-index thresholds must be in [0, 1] and critical <= warning.")
    if bool(config["use_rul_predictions_in_alerts"]):
        raise ValueError("RUL predictions in alerts are intentionally disabled in this phase.")
    if int(config["representative_timeline_count"]) <= 0:
        raise ValueError("representative_timeline_count must be positive.")
    for candidate in config["page_hinkley_candidates"]:
        if float(candidate["threshold"]) <= 0 or float(candidate["delta"]) < 0 or int(candidate["min_observations"]) <= 0:
            raise ValueError("Invalid Page-Hinkley candidate.")
        if candidate["direction"] not in {"increase", "decrease"}:
            raise ValueError("Invalid Page-Hinkley direction.")


def reproduce_phase2_frames(config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Reuse Phase 2 modules to reproduce scores with the same split and fitting rules."""
    phase2_config = load_phase2_config(resolve_project_path(config["phase2_config_path"], project_root()))
    model_train, validation, test, metadata = prepare_datasets(phase2_config)
    candidates, retained, reasons, _, _ = select_anomaly_features(
        model_train,
        include_cycle=bool(phase2_config["include_cycle_as_feature"]),
        configured_exclusions=list(phase2_config["features_to_exclude"]),
        near_constant_threshold=float(phase2_config["near_constant_threshold"]),
        correlation_threshold=float(phase2_config["correlation_threshold"]),
    )
    if not bool(phase2_config["include_cycle_as_feature"]):
        reasons[CYCLE_COLUMN] = "not configured as an anomaly-model feature"
    frames = {"model_train": model_train, "validation": validation, "test": test}
    frames, model_info = apply_health_and_detectors(frames, retained, phase2_config)
    for split, frame in frames.items():
        frame["split"] = "train" if split == "model_train" else split
    metadata.update(
        {
            "candidate_features": candidates,
            "retained_features": retained,
            "excluded_features": reasons,
            "phase2_model_info": model_info,
            "phase2_config": phase2_config,
        }
    )
    return frames, metadata


def calibrate_scores(frames: dict[str, pd.DataFrame], config: dict[str, Any]) -> dict[str, Any]:
    quantiles = config["calibration_quantiles"]
    metadata: dict[str, Any] = {
        "method": config["calibration_method"],
        "source_population": "healthy rows from model-training engines only",
        "detectors": {},
    }
    healthy_mask = frames["model_train"]["proxy_degradation_label"] == 0
    for detector in DETECTOR_ORDER:
        raw_col = RAW_SCORE_COLUMNS[detector]
        cal_col = CALIBRATED_SCORE_COLUMNS[detector]
        calibrator = ScoreCalibrator(
            method=config["calibration_method"],
            lower_quantile=float(quantiles["lower"]),
            upper_quantile=float(quantiles["upper"]),
            epsilon=float(config["epsilon"]),
            clip=bool(config.get("clip_calibrated_scores", True)),
        )
        calibrator.fit(frames["model_train"].loc[healthy_mask, raw_col])
        for frame in frames.values():
            frame[cal_col] = calibrator.transform(frame[raw_col])
        metadata["detectors"][detector] = {
            "raw_score_column": raw_col,
            "calibrated_score_column": cal_col,
            **calibrator.metadata(),
        }
    return metadata


def critical_recall(frame: pd.DataFrame, flag_col: str) -> float | None:
    critical = frame[frame["proxy_critical_label"] == 1]
    if critical.empty:
        return None
    return float(critical[flag_col].astype(bool).mean())


def percent(value: int | float | None, denominator: int | float | None) -> float | None:
    if value is None or denominator in {None, 0}:
        return None
    return float(value) / float(denominator)


def evaluate_frame_candidate(
    frame: pd.DataFrame,
    score_col: str,
    flag_col: str,
    candidate_name: str,
    candidate_kind: str,
    persistence_rule: dict,
    healthy_rul_threshold: float,
    critical_rul_threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    row = row_level_anomaly_metrics(
        frame["proxy_degradation_label"],
        frame[flag_col],
        frame[score_col],
    )
    persisted, persist_summary = apply_persistence_rule(
        frame,
        flag_column=flag_col,
        score_column=score_col,
        output_prefix="candidate",
        rule=persistence_rule,
    )
    onset_summary, engine = summarize_engine_onsets(
        persisted,
        detection_flag_column="candidate_persistent_alarm_state",
        method_name=candidate_name,
        split_name=str(frame["split"].iloc[0]),
        healthy_rul_threshold=healthy_rul_threshold,
        critical_rul_threshold=critical_rul_threshold,
    )
    transitions = int(persist_summary["number_of_alarm_transitions"].sum()) if not persist_summary.empty else 0
    detected = engine.get("detected_engines", 0)
    summary = {
        "candidate_name": candidate_name,
        "candidate_kind": candidate_kind,
        "score_column": score_col,
        "flag_column": flag_col,
        "precision": row["precision"],
        "recall": row["recall"],
        "f1": row["f1"],
        "pr_auc": row["pr_auc"],
        "roc_auc": row["roc_auc"],
        "specificity": row["specificity"],
        "healthy_region_false_positive_rate": row["false_positive_rate"],
        "critical_region_recall": critical_recall(frame, flag_col),
        "balanced_accuracy": row["balanced_accuracy"],
        "engine_detection_rate": engine["detection_rate"],
        "false_alarm_engine_count": engine["false_alarm_engine_count"],
        "missed_engine_count": engine["missed_engines"],
        "median_detection_delay": engine["median_detection_delay"],
        "median_lead_time": engine["median_lead_time"],
        "detected_before_60_fraction": percent(engine["detections_before_60_cycles_rul"], detected),
        "detected_before_30_fraction": percent(engine["detections_before_30_cycles_rul"], detected),
        "detected_engines": engine["detected_engines"],
        "engines_evaluated": engine["engines_evaluated"],
        "transition_count": transitions,
    }
    return summary, onset_summary


def add_utilities(row: dict[str, Any], profiles: dict[str, dict[str, float]]) -> dict[str, Any]:
    row_metrics = {
        "false_positive_rate": row.get("healthy_region_false_positive_rate"),
    }
    engine_metrics = {
        "engines_evaluated": row.get("engines_evaluated"),
        "detected_engines": row.get("detected_engines"),
        "missed_engines": row.get("missed_engine_count"),
        "false_alarm_engine_count": row.get("false_alarm_engine_count"),
        "detections_before_30_cycles_rul": None
        if row.get("detected_before_30_fraction") is None
        else row.get("detected_before_30_fraction") * row.get("detected_engines", 0),
        "detection_rate": row.get("engine_detection_rate"),
    }
    for profile, weights in profiles.items():
        row[f"utility_{profile}"] = compute_operating_utility(
            row_metrics,
            engine_metrics,
            int(row.get("transition_count") or 0),
            weights,
        )
    return row


def default_persistence_rule(config: dict[str, Any]) -> dict[str, Any]:
    value = int(config["default_persistence"]["consecutive"])
    return {"name": f"consecutive_{value}", "type": "consecutive", "k": value}


def threshold_operating_points(
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    validation = frames["validation"].copy()
    rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    rule = default_persistence_rule(config)
    for detector in DETECTOR_ORDER:
        native_row, _ = evaluate_frame_candidate(
            validation,
            RAW_SCORE_COLUMNS[detector],
            NATIVE_FLAG_COLUMNS[detector],
            candidate_name=f"{detector}_phase2_native_reference",
            candidate_kind="detector_native_reference",
            persistence_rule=rule,
            healthy_rul_threshold=float(config["healthy_rul_threshold"]),
            critical_rul_threshold=float(config["critical_rul_threshold"]),
        )
        native_row.update(
            {
                "detector": detector,
                "threshold": math.nan,
                "persistence_rule": rule["name"],
                "fusion_method": "",
                "voting_rule": "",
                "native_reference": True,
                "selection_eligible": False,
            }
        )
        rows.append(add_utilities(native_row, config["utility_profiles"]))
        score_col = CALIBRATED_SCORE_COLUMNS[detector]
        for threshold in config["candidate_detector_thresholds"]:
            flag_col = f"candidate_{detector}_{str(threshold).replace('.', '_')}_flag"
            validation[flag_col] = validation[score_col] >= float(threshold)
            row, _ = evaluate_frame_candidate(
                validation,
                score_col,
                flag_col,
                candidate_name=f"{detector}@{threshold}",
                candidate_kind="detector_threshold",
                persistence_rule=rule,
                healthy_rul_threshold=float(config["healthy_rul_threshold"]),
                critical_rul_threshold=float(config["critical_rul_threshold"]),
            )
            row.update(
                {
                    "detector": detector,
                    "threshold": float(threshold),
                    "persistence_rule": rule["name"],
                    "fusion_method": "",
                    "voting_rule": "",
                    "native_reference": False,
                    "selection_eligible": True,
                }
            )
            row = add_utilities(row, config["utility_profiles"])
            rows.append(row)
            ranking_rows.append(row)
    return pd.DataFrame(rows), ranking_rows


def ensemble_operating_points(
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    validation = frames["validation"].copy()
    rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    score_columns = CALIBRATED_SCORE_COLUMNS
    rule = default_persistence_rule(config)
    for method in config["candidate_fusion_methods"]:
        weight_sets = config["candidate_fusion_weights"] if method == "weighted_mean" else [None]
        for weight_idx, weights in enumerate(weight_sets):
            name = method if weights is None else f"weighted_mean_{weight_idx + 1}"
            validation[f"{name}_score"] = fuse_scores(validation, score_columns, method, weights)
            for threshold in config["candidate_detector_thresholds"]:
                flag_col = f"{name}_{str(threshold).replace('.', '_')}_flag"
                validation[flag_col] = validation[f"{name}_score"] >= float(threshold)
                row, _ = evaluate_frame_candidate(
                    validation,
                    f"{name}_score",
                    flag_col,
                    candidate_name=f"{name}@{threshold}",
                    candidate_kind="score_fusion",
                    persistence_rule=rule,
                    healthy_rul_threshold=float(config["healthy_rul_threshold"]),
                    critical_rul_threshold=float(config["critical_rul_threshold"]),
                )
                row.update(
                    {
                        "fusion_method": method,
                        "fusion_name": name,
                        "fusion_weights": "" if weights is None else json.dumps(weights, sort_keys=True),
                        "threshold": float(threshold),
                        "persistence_rule": rule["name"],
                        "detector": "",
                        "voting_rule": "",
                        "native_reference": False,
                        "selection_eligible": True,
                    }
                )
                row = add_utilities(row, config["utility_profiles"])
                rows.append(row)
                ranking_rows.append(row)
    voting_threshold = float(config["voting_detector_threshold"])
    flag_columns: dict[str, str] = {}
    for detector in DETECTOR_ORDER:
        flag_col = f"vote_{detector}_flag"
        validation[flag_col] = validation[CALIBRATED_SCORE_COLUMNS[detector]] >= voting_threshold
        flag_columns[detector] = flag_col
    validation["voting_score"] = voting_score(validation, flag_columns)
    for voting_rule in config["candidate_voting_rules"]:
        flag_col = f"voting_{voting_rule}_flag"
        validation[flag_col] = voting_flags(validation, flag_columns, voting_rule)
        row, _ = evaluate_frame_candidate(
            validation,
            "voting_score",
            flag_col,
            candidate_name=f"voting_{voting_rule}",
            candidate_kind="voting",
            persistence_rule=rule,
            healthy_rul_threshold=float(config["healthy_rul_threshold"]),
            critical_rul_threshold=float(config["critical_rul_threshold"]),
        )
        row.update(
            {
                "fusion_method": "",
                "fusion_name": "",
                "fusion_weights": "",
                "threshold": voting_threshold,
                "persistence_rule": rule["name"],
                "detector": "",
                "voting_rule": voting_rule,
                "native_reference": False,
                "selection_eligible": True,
            }
        )
        row = add_utilities(row, config["utility_profiles"])
        rows.append(row)
        ranking_rows.append(row)
    return pd.DataFrame(rows), ranking_rows


def persistence_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    rules = [
        {"name": f"consecutive_{value}", "type": "consecutive", "k": int(value)}
        for value in config["candidate_consecutive_persistence_values"]
    ]
    rules.extend(
        {"name": f"{int(rule['k'])}_of_{int(rule['n'])}", "type": "k_of_n", "k": int(rule["k"]), "n": int(rule["n"])}
        for rule in config["candidate_k_of_n_rules"]
    )
    rules.extend(
        {
            "name": f"score_duration_{int(rule['duration'])}_{rule['threshold']}",
            "type": "score_duration",
            "duration": int(rule["duration"]),
            "threshold": float(rule["threshold"]),
        }
        for rule in config.get("score_duration_rules", [])
    )
    return rules


def persistence_operating_points(
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    validation = frames["validation"].copy()
    score_columns = CALIBRATED_SCORE_COLUMNS
    validation["balanced_mean_score"] = fuse_scores(validation, score_columns, "mean")
    threshold = float(config["persistence_base_threshold"])
    validation["balanced_mean_flag"] = validation["balanced_mean_score"] >= threshold
    rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for rule in persistence_candidates(config):
        row, _ = evaluate_frame_candidate(
            validation,
            "balanced_mean_score",
            "balanced_mean_flag",
            candidate_name=f"balanced_mean_{rule['name']}",
            candidate_kind="persistence_rule",
            persistence_rule=rule,
            healthy_rul_threshold=float(config["healthy_rul_threshold"]),
            critical_rul_threshold=float(config["critical_rul_threshold"]),
        )
        row.update(
            {
                "fusion_method": "mean",
                "fusion_name": "balanced_mean",
                "threshold": threshold,
                "persistence_rule": rule["name"],
                "detector": "",
                "voting_rule": "",
                "native_reference": False,
                "selection_eligible": True,
            }
        )
        row = add_utilities(row, config["utility_profiles"])
        rows.append(row)
        ranking_rows.append(row)
    return pd.DataFrame(rows), ranking_rows


def page_hinkley_operating_points(frames: dict[str, pd.DataFrame], config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    validation = frames["validation"]
    for idx, candidate in enumerate(config["page_hinkley_candidates"], start=1):
        frame, _ = apply_page_hinkley_by_engine(
            validation.copy(),
            signal_column=candidate["primary_signal"],
            output_prefix=f"ph_{idx}",
            delta=float(candidate["delta"]),
            threshold=float(candidate["threshold"]),
            min_observations=int(candidate["min_observations"]),
            direction=candidate["direction"],
            reset_after_detection=bool(candidate.get("reset_after_detection", False)),
        )
        flag_col = f"ph_{idx}_change_flag"
        row, _ = evaluate_frame_candidate(
            frame,
            candidate["primary_signal"],
            flag_col,
            candidate_name=f"page_hinkley_{idx}",
            candidate_kind="page_hinkley",
            persistence_rule={"name": "consecutive_1", "type": "consecutive", "k": 1},
            healthy_rul_threshold=float(config["healthy_rul_threshold"]),
            critical_rul_threshold=float(config["critical_rul_threshold"]),
        )
        row.update(
            {
                "delta": float(candidate["delta"]),
                "threshold": float(candidate["threshold"]),
                "min_observations": int(candidate["min_observations"]),
                "direction": candidate["direction"],
                "healthy_region_alarm_frequency": row["healthy_region_false_positive_rate"],
                "native_reference": False,
                "selection_eligible": False,
                "standalone_conclusion": "supporting_monitor_signal"
                if row["false_alarm_engine_count"] > 0
                else "possible_standalone_alarm",
            }
        )
        row = add_utilities(row, config["utility_profiles"])
        rows.append(row)
    return pd.DataFrame(rows)


def selected_candidate_from_ranking(ranking: pd.DataFrame, profile: str) -> dict[str, Any]:
    if ranking.empty:
        raise ValueError("No operating points to rank.")
    utility_col = f"utility_{profile}"
    selected = ranking.sort_values(utility_col, ascending=False).iloc[0].to_dict()
    return selected


def apply_selected_candidate(
    frame: pd.DataFrame,
    selected: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    result = frame.copy()
    kind = selected["candidate_kind"]
    threshold = float(selected["threshold"])
    score_col = "selected_ensemble_score"
    flag_col = "selected_raw_anomaly_flag"
    weights = None
    voting_rule = selected.get("voting_rule") or ""
    fusion_method = selected.get("fusion_method") or ""
    if kind == "detector_threshold":
        detector = selected["detector"]
        result[score_col] = result[CALIBRATED_SCORE_COLUMNS[detector]]
    elif kind in {"score_fusion", "persistence_rule"}:
        method = fusion_method or "mean"
        if method == "weighted_mean":
            weights = json.loads(selected["fusion_weights"])
        result[score_col] = fuse_scores(result, CALIBRATED_SCORE_COLUMNS, method, weights)
    elif kind == "voting":
        detector_flag_cols = {}
        for detector in DETECTOR_ORDER:
            col = f"selected_vote_{detector}_flag"
            result[col] = result[CALIBRATED_SCORE_COLUMNS[detector]] >= threshold
            detector_flag_cols[detector] = col
        result[score_col] = voting_score(result, detector_flag_cols)
        result[flag_col] = voting_flags(result, detector_flag_cols, voting_rule)
    else:
        raise ValueError(f"Unsupported selected candidate kind: {kind}")
    if kind != "voting":
        result[flag_col] = result[score_col] >= threshold

    rule_name = selected.get("persistence_rule") or default_persistence_rule(config)["name"]
    rule = next((item for item in persistence_candidates(config) if item["name"] == rule_name), default_persistence_rule(config))
    result, persistence_summary = apply_persistence_rule(
        result,
        flag_column=flag_col,
        score_column=score_col,
        output_prefix="selected",
        rule=rule,
    )
    hysteresis = config["selected_hysteresis"]
    result, hysteresis_summary = apply_hysteresis_alert(
        result,
        score_column=score_col,
        output_prefix="selected",
        enter_threshold=float(hysteresis["enter_threshold"]),
        exit_threshold=float(hysteresis["exit_threshold"]),
        min_enter_duration=int(hysteresis["min_enter_duration"]),
        min_clear_duration=int(hysteresis["min_clear_duration"]),
    )
    result = assign_operational_alert_levels(
        result,
        score_column=score_col,
        persistent_column="selected_persistent_alarm_state",
        health_column="smoothed_health_index",
        output_column="operational_alert_level",
        thresholds=config["operational_alert_thresholds"],
    )
    details = {
        "kind": kind,
        "threshold": threshold,
        "fusion_method": fusion_method,
        "fusion_weights": weights,
        "voting_rule": voting_rule,
        "persistence_rule": rule,
        "hysteresis": hysteresis,
    }
    alert_summary = persistence_summary.merge(hysteresis_summary, on=UNIT_COLUMN, how="outer")
    return result, details, alert_summary


def evaluate_selected(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    row = row_level_anomaly_metrics(
        frame["proxy_degradation_label"],
        frame["selected_raw_anomaly_flag"],
        frame["selected_ensemble_score"],
    )
    onset_summary, engine = summarize_engine_onsets(
        frame,
        detection_flag_column="selected_persistent_alarm_state",
        method_name="selected_operating_point",
        split_name=str(frame["split"].iloc[0]),
        healthy_rul_threshold=float(config["healthy_rul_threshold"]),
        critical_rul_threshold=float(config["critical_rul_threshold"]),
    )
    transitions = alert_transition_metrics(frame, "selected_alert_state", "operational_alert_level")
    return {"row_level": row, "engine_level": engine, "alert_transitions": transitions}, onset_summary


def health_generalization(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split in ["validation", "test"]:
        frame = frames[split]
        for unit_id, group in frame.sort_values([UNIT_COLUMN, CYCLE_COLUMN]).groupby(UNIT_COLUMN):
            corr = group[["smoothed_health_index", "true_rul_uncapped"]].corr(method="spearman").iloc[0, 1]
            if pd.isna(corr):
                category = "undefined"
            elif corr < 0:
                category = "negative"
            elif corr < 0.3:
                category = "weak"
            elif corr < 0.6:
                category = "moderate"
            else:
                category = "strong"
            rows.append(
                {
                    "split": split,
                    "unit_id": int(unit_id),
                    "trajectory_length": int(len(group)),
                    "initial_health_index": float(group["smoothed_health_index"].iloc[0]),
                    "final_health_index": float(group["smoothed_health_index"].iloc[-1]),
                    "health_index_delta": float(group["smoothed_health_index"].iloc[-1] - group["smoothed_health_index"].iloc[0]),
                    "health_index_raw_min": float(group["health_index_raw"].min()),
                    "health_index_raw_max": float(group["health_index_raw"].max()),
                    "health_index_scaled_min": float(group["health_index_scaled"].min()),
                    "health_index_scaled_max": float(group["health_index_scaled"].max()),
                    "initial_true_rul": float(group["true_rul_uncapped"].iloc[0]),
                    "final_true_rul": float(group["true_rul_uncapped"].iloc[-1]),
                    "observed_rul_span": float(group["true_rul_uncapped"].iloc[0] - group["true_rul_uncapped"].iloc[-1]),
                    "spearman_correlation": None if pd.isna(corr) else float(corr),
                    "correlation_category": category,
                    "orientation_consistent_with_rul": bool(False if pd.isna(corr) else corr >= 0),
                }
            )
    return pd.DataFrame(rows)


def make_figures(
    output_dir: Path,
    threshold_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
    persistence_df: pd.DataFrame,
    ph_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    health_df: pd.DataFrame,
    selected_frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> list[str]:
    figures = []
    figures_dir = output_dir / "figures"
    timelines_dir = output_dir / "engine_timelines"
    figures_dir.mkdir(parents=True, exist_ok=True)
    timelines_dir.mkdir(parents=True, exist_ok=True)

    def save_current(path: Path) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        figures.append(str(path))

    plt.figure(figsize=(8, 5))
    for detector, group in threshold_df.groupby("detector"):
        plt.plot(group["threshold"], group["precision"], marker="o", label=f"{detector} precision")
        plt.plot(group["threshold"], group["recall"], marker="x", linestyle="--", label=f"{detector} recall")
    plt.title("Threshold Versus Precision/Recall")
    plt.xlabel("Calibrated threshold")
    plt.ylabel("Metric")
    plt.legend(fontsize=7)
    save_current(figures_dir / "threshold_precision_recall_tradeoff.png")

    plt.figure(figsize=(8, 5))
    for detector, group in threshold_df.groupby("detector"):
        plt.plot(group["threshold"], group["healthy_region_false_positive_rate"], marker="o", label=detector)
    plt.title("Threshold Versus Healthy-Region FPR")
    plt.xlabel("Calibrated threshold")
    plt.ylabel("Healthy-region FPR")
    plt.legend()
    save_current(figures_dir / "threshold_healthy_fpr.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(ranking_df["false_alarm_engine_count"] / ranking_df["engines_evaluated"], ranking_df["engine_detection_rate"], alpha=0.75)
    plt.title("Detection Rate Versus False-Alarm Engine Rate")
    plt.xlabel("False-alarm engine rate")
    plt.ylabel("Engine detection rate")
    save_current(figures_dir / "detection_vs_false_alarm_rate.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(ranking_df["false_alarm_engine_count"] / ranking_df["engines_evaluated"], ranking_df["median_lead_time"], alpha=0.75)
    plt.title("Lead Time Versus False-Alarm Engine Rate")
    plt.xlabel("False-alarm engine rate")
    plt.ylabel("Median lead time")
    save_current(figures_dir / "lead_time_vs_false_alarm_rate.png")

    plt.figure(figsize=(9, 5))
    fusion = ensemble_df[ensemble_df["candidate_kind"] == "score_fusion"]
    means = fusion.groupby("fusion_name")["f1"].max().sort_values(ascending=False)
    plt.bar(means.index, means.values)
    plt.title("Ensemble Method Comparison")
    plt.ylabel("Best validation F1")
    plt.xticks(rotation=25, ha="right")
    save_current(figures_dir / "ensemble_method_comparison.png")

    plt.figure(figsize=(7, 5))
    voting = ensemble_df[ensemble_df["candidate_kind"] == "voting"]
    plt.bar(voting["voting_rule"], voting["f1"])
    plt.title("Voting Rule Comparison")
    plt.ylabel("Validation F1")
    save_current(figures_dir / "voting_rule_comparison.png")

    plt.figure(figsize=(9, 5))
    plt.bar(persistence_df["persistence_rule"], persistence_df["engine_detection_rate"])
    plt.title("Persistence Rule Comparison")
    plt.ylabel("Engine detection rate")
    plt.xticks(rotation=25, ha="right")
    save_current(figures_dir / "persistence_rule_comparison.png")

    plt.figure(figsize=(8, 5))
    profiles = [col for col in ranking_df.columns if col.startswith("utility_")]
    best = [ranking_df[col].max() for col in profiles]
    plt.bar([col.replace("utility_", "") for col in profiles], best)
    plt.title("Operational Profile Comparison")
    plt.ylabel("Best validation utility")
    save_current(figures_dir / "operational_profile_comparison.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(ph_df["false_alarm_engine_count"], ph_df["engine_detection_rate"])
    for _, row in ph_df.iterrows():
        plt.annotate(str(row["candidate_name"]), (row["false_alarm_engine_count"], row["engine_detection_rate"]), fontsize=7)
    plt.title("Page-Hinkley Sensitivity Versus False Alarms")
    plt.xlabel("False-alarm engines")
    plt.ylabel("Engine detection rate")
    save_current(figures_dir / "page_hinkley_sensitivity_false_alarm.png")

    plt.figure(figsize=(8, 5))
    selected_val = selected_frames["validation"]
    raw_transitions = selected_val.groupby(UNIT_COLUMN)["selected_raw_anomaly_flag"].apply(
        lambda s: sum(a != b for a, b in zip(s.astype(bool), s.astype(bool).iloc[1:]))
    )
    alert_transitions = selected_val.groupby(UNIT_COLUMN)["selected_alert_state"].apply(
        lambda s: sum(a != b for a, b in zip(s.astype(bool), s.astype(bool).iloc[1:]))
    )
    plt.hist([raw_transitions, alert_transitions], bins=10, label=["raw flag", "hysteresis alert"], alpha=0.7)
    plt.title("Hysteresis Transition Comparison")
    plt.xlabel("Transitions per validation engine")
    plt.ylabel("Engine count")
    plt.legend()
    save_current(figures_dir / "hysteresis_transition_comparison.png")

    plt.figure(figsize=(8, 5))
    for split, group in health_df.groupby("split"):
        plt.hist(group["spearman_correlation"].dropna(), bins=20, alpha=0.5, label=split)
    plt.title("Per-Engine Health-Index Correlation Distribution")
    plt.xlabel("Spearman correlation")
    plt.ylabel("Engine count")
    plt.legend()
    save_current(figures_dir / "health_correlation_distribution.png")

    plt.figure(figsize=(8, 5))
    for split, frame in selected_frames.items():
        if split in {"validation", "test"}:
            plt.hist(frame["smoothed_health_index"], bins=30, alpha=0.45, label=split)
    plt.title("Validation Versus Test Health-Index Distribution")
    plt.xlabel("Smoothed health index")
    plt.ylabel("Cycle count")
    plt.legend()
    save_current(figures_dir / "validation_test_health_distribution.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(health_df["trajectory_length"], health_df["spearman_correlation"], alpha=0.7)
    plt.title("Health Correlation Versus Trajectory Length")
    plt.xlabel("Observed trajectory length")
    plt.ylabel("Per-engine Spearman correlation")
    save_current(figures_dir / "health_correlation_vs_length.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(health_df["initial_health_index"], health_df["final_health_index"], alpha=0.7)
    plt.title("Initial Versus Final Health Index")
    plt.xlabel("Initial health index")
    plt.ylabel("Final health index")
    save_current(figures_dir / "initial_vs_final_health_index.png")

    test_health = health_df[health_df["split"] == "test"].dropna(subset=["spearman_correlation"])
    chosen_units = []
    if not test_health.empty:
        chosen_units.append(int(test_health.sort_values("spearman_correlation", ascending=False)["unit_id"].iloc[0]))
        chosen_units.append(int(test_health.sort_values("spearman_correlation", ascending=True)["unit_id"].iloc[0]))
    for unit_id in sorted(set(chosen_units)):
        group = selected_frames["test"][selected_frames["test"][UNIT_COLUMN] == unit_id]
        plt.figure(figsize=(9, 5))
        plt.plot(group[CYCLE_COLUMN], group["smoothed_health_index"], label="smoothed health index")
        plt.plot(group[CYCLE_COLUMN], group["true_rul_uncapped"] / max(group["true_rul_uncapped"].max(), 1), label="true RUL normalized")
        plt.title(f"Representative Health Trajectory: Test Engine {unit_id}")
        plt.xlabel("Cycle")
        plt.legend()
        save_current(figures_dir / f"health_trajectory_engine_{unit_id:03d}.png")

    def timeline_categories(frame: pd.DataFrame) -> list[tuple[int, str]]:
        categories: list[tuple[int, str]] = []
        seen: set[str] = set()
        for unit_id, group in frame.sort_values([UNIT_COLUMN, CYCLE_COLUMN]).groupby(UNIT_COLUMN):
            alarms = group[group["selected_persistent_alarm_state"].astype(bool)]
            if alarms.empty:
                category = "missed degradation"
            else:
                first_rul = float(alarms["true_rul_uncapped"].iloc[0])
                if first_rul > float(config["healthy_rul_threshold"]):
                    category = "healthy-region false alarm"
                elif first_rul <= float(config["critical_rul_threshold"]):
                    category = "late critical detection"
                else:
                    category = "early warning"
            if category not in seen:
                categories.append((int(unit_id), category))
                seen.add(category)
            if len(categories) >= int(config["representative_timeline_count"]):
                break
        for unit_id in sorted(frame[UNIT_COLUMN].unique()):
            if len(categories) >= int(config["representative_timeline_count"]):
                break
            if int(unit_id) not in {item[0] for item in categories}:
                categories.append((int(unit_id), "additional test engine"))
        return categories

    level_map = {"NORMAL": 0, "MONITOR": 1, "WARNING": 2, "CRITICAL": 3}
    for unit_id, category in timeline_categories(selected_frames["test"]):
        group = selected_frames["test"][selected_frames["test"][UNIT_COLUMN] == unit_id].sort_values(CYCLE_COLUMN)
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        for detector in DETECTOR_ORDER:
            axes[0].plot(group[CYCLE_COLUMN], group[CALIBRATED_SCORE_COLUMNS[detector]], label=detector)
        axes[0].plot(group[CYCLE_COLUMN], group["selected_ensemble_score"], color="black", linewidth=1.6, label="selected ensemble")
        axes[0].axhline(float(config["selected_hysteresis"]["enter_threshold"]), color="red", linestyle="--", label="enter")
        axes[0].axhline(float(config["selected_hysteresis"]["exit_threshold"]), color="green", linestyle="--", label="exit")
        axes[0].set_ylabel("Calibrated scores")
        axes[0].legend(fontsize=7, ncol=2)
        axes[1].plot(group[CYCLE_COLUMN], group["smoothed_health_index"], label="health index", color="tab:green")
        axes[1].set_ylabel("Health index")
        axes[1].legend(fontsize=8)
        axes[2].step(group[CYCLE_COLUMN], group["selected_persistent_alarm_state"].astype(int), where="post", label="persistent")
        axes[2].step(group[CYCLE_COLUMN], group["selected_alert_state"].astype(int), where="post", label="hysteresis")
        level_values = group["operational_alert_level"].map(level_map).fillna(0)
        axes[2].step(group[CYCLE_COLUMN], level_values, where="post", label="alert level", alpha=0.75)
        axes[2].set_ylabel("Alarm / level")
        axes[2].set_yticks([0, 1, 2, 3])
        axes[2].set_yticklabels(["NORMAL", "MONITOR", "WARNING", "CRITICAL"])
        axes[2].legend(fontsize=8)
        axes[3].plot(group[CYCLE_COLUMN], group["true_rul_uncapped"], label="true RUL")
        onset = group.loc[group["proxy_degradation_label"] == 1, CYCLE_COLUMN]
        critical = group.loc[group["proxy_critical_label"] == 1, CYCLE_COLUMN]
        if len(onset):
            axes[3].axvline(onset.iloc[0], color="orange", linestyle="--", label="proxy degradation")
        if len(critical):
            axes[3].axvline(critical.iloc[0], color="red", linestyle="--", label="proxy critical")
        axes[3].set_ylabel("RUL cycles")
        axes[3].set_xlabel("Cycle")
        axes[3].legend(fontsize=8)
        fig.suptitle(f"Selected Alert Timeline: Test Engine {unit_id} ({category})")
        path = timelines_dir / f"test_engine_{int(unit_id):03d}_alert_timeline.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(str(path))
    return figures


def write_results_note(path: Path, result: dict[str, Any], config_path: Path) -> None:
    selected = result["selected_operating_point"]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# FD001 Anomaly Calibration Results\n\n")
        handle.write("Selection is validation-only. Final test metrics are transparent evaluation, not selection input.\n\n")
        handle.write(f"- Python interpreter: `{result['python_executable']}`\n")
        handle.write(f"- Python version: `{result['python_version']}`\n")
        handle.write(f"- Dataset dimensions: train `{result['train_shape']}`, test `{result['test_shape']}`\n")
        handle.write(f"- Engine split: model-train `{result['model_train_engine_count']}`, validation `{result['validation_engine_count']}`, test `{result['test_engine_count']}`\n")
        handle.write(f"- Calibration method: `{result['calibration_method']}`\n")
        handle.write(f"- Selected profile: `{selected['selected_profile']}`\n")
        handle.write(f"- Selected candidate: `{selected['candidate_name']}`\n")
        handle.write(f"- Selected threshold: `{selected['threshold']}`\n")
        handle.write(f"- Selected persistence rule: `{selected['persistence_rule']}`\n")
        handle.write(f"- Selected hysteresis: `{selected['hysteresis']}`\n")
        handle.write(f"- Runtime seconds: `{result['runtime_seconds']:.3f}`\n\n")
        handle.write("## Calibration Statistics\n\n")
        for detector, stats in result["calibration_statistics"]["detectors"].items():
            handle.write(
                f"- `{detector}`: healthy rows `{stats['healthy_score_count']}`, "
                f"median `{stats['median']}`, lower `{stats['lower_value']}`, upper `{stats['upper_value']}`\n"
            )
        handle.write("\n## Validation Operating-Point Selection\n\n")
        handle.write(
            "Native Phase 2 detector thresholds are retained as reference rows in "
            "`threshold_operating_points.csv`; validation selection uses the bounded "
            "Phase 2B candidate table only.\n\n"
        )
        for row in result["top_validation_candidates"]:
            handle.write(
                f"- `{row['candidate_name']}` utility `{row['selection_utility']}` "
                f"kind `{row['candidate_kind']}` threshold `{row['threshold']}`\n"
            )
        handle.write("\n## Comparison Artifacts\n\n")
        handle.write("- Threshold comparisons: `threshold_operating_points.csv`\n")
        handle.write("- Ensemble and voting comparisons: `ensemble_operating_points.csv`\n")
        handle.write("- Persistence comparisons: `persistence_operating_points.csv`\n")
        handle.write("- Page-Hinkley reassessment: `page_hinkley_operating_points.csv`\n")
        handle.write("- Utility-profile ranking: `validation_operating_point_ranking.csv`\n\n")
        handle.write("## Validation Metrics\n\n")
        handle.write(f"`{result['validation_metrics']}`\n\n")
        handle.write("## Final Test Metrics\n\n")
        handle.write(f"`{result['test_metrics']}`\n\n")
        handle.write("## Page-Hinkley Reassessment\n\n")
        handle.write(f"`{result['page_hinkley_conclusion']}`\n\n")
        handle.write("## Health-Index Generalization Findings\n\n")
        handle.write(f"`{result['health_generalization_findings']}`\n\n")
        handle.write("## Generated Files\n\n")
        for item in result["generated_files"]:
            handle.write(f"- `{item}`\n")
        handle.write("\n## Warnings and Limitations\n\n")
        for warning in result["warnings"]:
            handle.write(f"- {warning}\n")
        handle.write("\n## Exact Reproduction Command\n\n")
        handle.write("```powershell\n")
        handle.write('$env:PYTHONPATH = ".\\src"\n')
        handle.write(
            "python -m aeroguard.pipelines.calibrate_fd001_anomaly_alerts "
            f'--config "{config_path.as_posix()}"\n'
        )
        handle.write("```\n")


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    start = time.perf_counter()
    root = project_root()
    config_path = Path(config_path)
    config = load_config(config_path)
    output_dir = resolve_project_path(config["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "engine_timelines").mkdir(parents=True, exist_ok=True)

    frames, metadata = reproduce_phase2_frames(config)
    calibration_metadata = calibrate_scores(frames, config)
    write_json(output_dir / "score_calibration.json", calibration_metadata)

    threshold_df, threshold_rank = threshold_operating_points(frames, config)
    ensemble_df, ensemble_rank = ensemble_operating_points(frames, config)
    persistence_df, persistence_rank = persistence_operating_points(frames, config)
    ph_df = page_hinkley_operating_points(frames, config)

    ranking_rows = threshold_rank + ensemble_rank + persistence_rank
    ranking_df = pd.DataFrame(ranking_rows)
    selected = selected_candidate_from_ranking(ranking_df, config["primary_utility_profile"])
    selected_validation, selected_details, validation_alert_summary = apply_selected_candidate(
        frames["validation"],
        selected,
        config,
    )
    selected_test, _, test_alert_summary = apply_selected_candidate(frames["test"], selected, config)
    selected_train, _, _ = apply_selected_candidate(frames["model_train"], selected, config)
    selected_frames = {"train": selected_train, "validation": selected_validation, "test": selected_test}
    validation_metrics, validation_onset = evaluate_selected(selected_validation, config)
    test_metrics, test_onset = evaluate_selected(selected_test, config)
    engine_alert_summary = pd.concat(
        [
            validation_alert_summary.assign(split="validation"),
            test_alert_summary.assign(split="test"),
        ],
        ignore_index=True,
    )
    health_df = health_generalization({"validation": selected_validation, "test": selected_test})
    health_findings = {
        "validation_median_correlation": float(health_df.loc[health_df["split"] == "validation", "spearman_correlation"].median()),
        "test_median_correlation": float(health_df.loc[health_df["split"] == "test", "spearman_correlation"].median()),
        "validation_correlation_categories": health_df[health_df["split"] == "validation"]["correlation_category"].value_counts().to_dict(),
        "test_correlation_categories": health_df[health_df["split"] == "test"]["correlation_category"].value_counts().to_dict(),
        "test_negative_or_weak_engines": int(
            health_df[(health_df["split"] == "test") & (health_df["correlation_category"].isin(["negative", "weak"]))].shape[0]
        ),
        "interpretation": "Diagnostic only; final test health-index analysis did not refit PCA or thresholds.",
    }
    ph_conclusion = {
        "validation_best_detection_rate": float(ph_df["engine_detection_rate"].max()) if not ph_df.empty else None,
        "validation_min_false_alarm_engines": int(ph_df["false_alarm_engine_count"].min()) if not ph_df.empty else None,
        "conclusion": "Page-Hinkley is retained as a high-sensitivity early-monitor signal, not the selected standalone operational alarm when false alarms are excessive.",
    }

    threshold_df.to_csv(output_dir / "threshold_operating_points.csv", index=False)
    ensemble_df.to_csv(output_dir / "ensemble_operating_points.csv", index=False)
    persistence_df.to_csv(output_dir / "persistence_operating_points.csv", index=False)
    ph_df.to_csv(output_dir / "page_hinkley_operating_points.csv", index=False)
    ranking_df.to_csv(output_dir / "validation_operating_point_ranking.csv", index=False)
    health_df.to_csv(output_dir / "health_index_generalization.csv", index=False)
    engine_alert_summary.to_csv(output_dir / "engine_alert_summary.csv", index=False)
    cycle_cols = [
        "split",
        UNIT_COLUMN,
        CYCLE_COLUMN,
        "true_rul_uncapped",
        "proxy_degradation_label",
        "proxy_critical_label",
        *RAW_SCORE_COLUMNS.values(),
        *CALIBRATED_SCORE_COLUMNS.values(),
        "selected_ensemble_score",
        "selected_raw_anomaly_flag",
        "selected_persistent_alarm_state",
        "selected_alert_state",
        "operational_alert_level",
        "smoothed_health_index",
        "page_hinkley_change_flag",
    ]
    cycle_level = pd.concat(selected_frames.values(), ignore_index=True)
    cycle_level[[col for col in cycle_cols if col in cycle_level.columns]].to_csv(
        output_dir / "cycle_level_alerts.csv",
        index=False,
    )
    write_json(output_dir / "validation_metrics.json", validation_metrics)
    write_json(output_dir / "test_metrics.json", test_metrics)
    selected_payload = {
        "selected_profile": config["primary_utility_profile"],
        "candidate_name": selected["candidate_name"],
        "candidate_kind": selected["candidate_kind"],
        "score_calibration_method": config["calibration_method"],
        "threshold": selected["threshold"],
        "fusion_method": selected_details.get("fusion_method"),
        "fusion_weights": selected_details.get("fusion_weights"),
        "voting_rule": selected_details.get("voting_rule"),
        "persistence_rule": selected_details["persistence_rule"],
        "hysteresis": selected_details["hysteresis"],
        "validation_metrics": validation_metrics,
        "selection_utility": selected[f"utility_{config['primary_utility_profile']}"],
        "reason_for_selection": f"Highest validation utility under {config['primary_utility_profile']} profile.",
    }
    write_json(output_dir / "selected_operating_points.json", selected_payload)
    figures = make_figures(
        output_dir,
        threshold_df,
        ensemble_df,
        persistence_df,
        ph_df,
        ranking_df,
        health_df,
        selected_frames,
        config,
    )

    runtime = time.perf_counter() - start
    primary_utility_col = f"utility_{config['primary_utility_profile']}"
    top_validation_candidates = []
    for _, row in ranking_df.sort_values(primary_utility_col, ascending=False).head(5).iterrows():
        top_validation_candidates.append(
            {
                "candidate_name": row["candidate_name"],
                "candidate_kind": row["candidate_kind"],
                "threshold": row.get("threshold"),
                "persistence_rule": row.get("persistence_rule"),
                "engine_detection_rate": row.get("engine_detection_rate"),
                "false_alarm_engine_count": row.get("false_alarm_engine_count"),
                "missed_engine_count": row.get("missed_engine_count"),
                "median_lead_time": row.get("median_lead_time"),
                "selection_utility": row.get(primary_utility_col),
            }
        )
    results_note = resolve_project_path(config.get("results_note_path", "notes/fd001_anomaly_calibration_results.md"), root)
    run_summary_path = output_dir / "run_summary.json"
    generated_files = [
        str(output_dir / "score_calibration.json"),
        str(output_dir / "threshold_operating_points.csv"),
        str(output_dir / "ensemble_operating_points.csv"),
        str(output_dir / "persistence_operating_points.csv"),
        str(output_dir / "page_hinkley_operating_points.csv"),
        str(output_dir / "validation_operating_point_ranking.csv"),
        str(output_dir / "selected_operating_points.json"),
        str(output_dir / "validation_metrics.json"),
        str(output_dir / "test_metrics.json"),
        str(output_dir / "engine_alert_summary.csv"),
        str(output_dir / "cycle_level_alerts.csv"),
        str(output_dir / "health_index_generalization.csv"),
        *figures,
        str(results_note),
        str(run_summary_path),
    ]
    result = {
        **{k: metadata[k] for k in ["dataset_dir", "train_shape", "test_shape", "train_engine_count", "model_train_engine_count", "validation_engine_count", "test_engine_count"]},
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "calibration_method": config["calibration_method"],
        "calibration_statistics": calibration_metadata,
        "candidate_operating_points_evaluated": int(len(threshold_df) + len(ensemble_df) + len(persistence_df) + len(ph_df)),
        "phase2_native_reference_points": int(threshold_df.get("native_reference", pd.Series(dtype=bool)).astype(bool).sum()),
        "top_validation_candidates": top_validation_candidates,
        "selected_operating_point": selected_payload,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "page_hinkley_conclusion": ph_conclusion,
        "health_generalization_findings": health_findings,
        "retained_features": metadata["retained_features"],
        "excluded_features": metadata["excluded_features"],
        "generated_files": generated_files,
        "runtime_seconds": runtime,
        "warnings": [
            "Proxy labels are RUL-threshold evaluation proxies, not certified physical anomaly truth.",
            "Operating-point selection uses validation data only; test metrics are final evaluation only.",
            "RUL predictions are not used in operational alert levels in this phase.",
        ],
    }
    write_results_note(results_note, result, config_path)
    write_json(run_summary_path, result)
    print(json.dumps(_json_ready(result), indent=2, allow_nan=False))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate FD001 anomaly scores and operational alerts.")
    parser.add_argument("--config", required=True, help="Path to Phase 2B YAML configuration.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
