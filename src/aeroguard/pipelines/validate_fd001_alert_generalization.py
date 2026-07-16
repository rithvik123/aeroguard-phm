"""Repeated FD001 alert-policy validation and FD003 transfer evaluation."""

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
from sklearn.preprocessing import StandardScaler

from aeroguard.anomaly.isolation_forest import IsolationForestAnomalyDetector
from aeroguard.anomaly.one_class_svm import OneClassSVMAnomalyDetector
from aeroguard.anomaly.pca_reconstruction import PCAReconstructionAnomalyDetector
from aeroguard.anomaly.policy_registry import (
    CALIBRATED_SCORE_COLUMNS,
    apply_alert_policy,
    validate_policy_registry,
)
from aeroguard.anomaly.score_calibration import ScoreCalibrator
from aeroguard.data.columns import BASE_FEATURE_COLUMNS, CYCLE_COLUMN, UNIT_COLUMN
from aeroguard.data.loader import load_cmapss_dataset, read_cmapss_table, read_rul_file
from aeroguard.data.targets import add_training_rul_targets
from aeroguard.data.validation import validate_test_rul_alignment
from aeroguard.evaluation.alert_metrics import alert_transition_metrics
from aeroguard.evaluation.anomaly_metrics import row_level_anomaly_metrics, summarize_engine_onsets
from aeroguard.evaluation.bootstrap import alert_engine_metric_functions, bootstrap_engine_metrics
from aeroguard.evaluation.domain_shift import (
    distribution_summary,
    feature_shift_table,
    trajectory_summary,
)
from aeroguard.evaluation.generalization_metrics import (
    classify_generalization,
    compute_profile_utility,
    select_locked_policy,
    summarize_policy_folds,
)
from aeroguard.evaluation.group_cross_validation import (
    GroupFold,
    repeated_group_kfold_splits,
    validate_group_folds,
)
from aeroguard.features.preprocessing import audit_features
from aeroguard.health.pca_health_index import PCAHealthIndex
from aeroguard.health.smoothing import smooth_by_engine
from aeroguard.onset.onset_detection import (
    add_proxy_labels,
    apply_page_hinkley_by_engine,
    derive_test_true_rul_trajectory,
)
from aeroguard.pipelines.train_fd001_baseline import project_root, resolve_project_path
from aeroguard.pipelines.train_fd001_health_anomaly import load_config as load_phase2_config
from aeroguard.pipelines.train_fd001_health_anomaly import select_anomaly_features


RAW_SCORE_COLUMNS = {
    "pca_reconstruction": "pca_anomaly_score",
    "isolation_forest": "isolation_forest_score",
    "one_class_svm": "one_class_svm_score",
}

REQUIRED_CONFIG_KEYS = {
    "dataset_dir",
    "fd001_subset",
    "fd003_subset",
    "baseline_config_path",
    "phase2_config_path",
    "phase2b_config_path",
    "random_seed",
    "group_cross_validation_folds",
    "group_cross_validation_repeats",
    "group_cross_validation_seeds",
    "healthy_rul_threshold",
    "critical_rul_threshold",
    "score_calibration",
    "candidate_policy_registry",
    "maximum_policy_count",
    "operational_profiles",
    "primary_operational_profile",
    "feasibility_constraints",
    "utility_variability_penalty",
    "alert_unstable_transition_threshold",
    "bootstrap_samples",
    "confidence_level",
    "bootstrap_seed",
    "health_index_correlation_categories",
    "generalization_classification_criteria",
    "operational_alert_thresholds",
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
    if str(config["fd001_subset"]).upper() != "FD001":
        raise ValueError("fd001_subset must be FD001.")
    if str(config["fd003_subset"]).upper() != "FD003":
        raise ValueError("fd003_subset must be FD003.")
    folds = int(config["group_cross_validation_folds"])
    repeats = int(config["group_cross_validation_repeats"])
    if folds < 2:
        raise ValueError("group_cross_validation_folds must be at least 2.")
    if repeats < 1:
        raise ValueError("group_cross_validation_repeats must be positive.")
    seeds = list(config["group_cross_validation_seeds"])
    if len(seeds) != repeats:
        raise ValueError("group_cross_validation_seeds length must equal repeats.")
    if float(config["healthy_rul_threshold"]) <= float(config["critical_rul_threshold"]):
        raise ValueError("healthy_rul_threshold must be greater than critical_rul_threshold.")
    if int(config["maximum_policy_count"]) <= 0:
        raise ValueError("maximum_policy_count must be positive.")
    if int(config["bootstrap_samples"]) <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    if not 0 < float(config["confidence_level"]) < 1:
        raise ValueError("confidence_level must be in (0, 1).")
    if float(config["utility_variability_penalty"]) < 0:
        raise ValueError("utility_variability_penalty must be non-negative.")
    if int(config["representative_timeline_count"]) <= 0:
        raise ValueError("representative_timeline_count must be positive.")
    for key in ["dataset_dir", "baseline_config_path", "phase2_config_path", "phase2b_config_path"]:
        path = resolve_project_path(config[key], root)
        if not path.exists():
            raise FileNotFoundError(f"Required path not found for {key}: {path}")
    dataset_dir = resolve_project_path(config["dataset_dir"], root)
    for subset in [config["fd001_subset"], config["fd003_subset"]]:
        for name in [f"test_{str(subset).upper()}.txt", f"RUL_{str(subset).upper()}.txt"]:
            if not (dataset_dir / name).exists():
                raise FileNotFoundError(f"Required dataset file missing: {dataset_dir / name}")
    output_dir = resolve_project_path(config["output_dir"], root)
    lowered = str(output_dir).lower()
    if "\\references\\" in lowered or "\\extracted-code\\" in lowered:
        raise ValueError("Output path must not be inside read-only reference directories.")
    calibration = config["score_calibration"]
    if calibration["method"] not in {"empirical_percentile", "robust_z", "quantile"}:
        raise ValueError("Invalid score calibration method.")
    if not 0 <= float(calibration["lower_quantile"]) < float(calibration["upper_quantile"]) <= 1:
        raise ValueError("Invalid score calibration quantiles.")
    if float(calibration["epsilon"]) <= 0:
        raise ValueError("Calibration epsilon must be positive.")
    registry = validate_policy_registry(
        list(config["candidate_policy_registry"]),
        maximum_policy_count=int(config["maximum_policy_count"]),
        operational_profiles=dict(config["operational_profiles"]),
    )
    if not registry:
        raise ValueError("candidate_policy_registry must not be empty.")
    mismatched_calibration = [
        policy["policy_id"]
        for policy in registry
        if str(policy["calibration_method"]) != str(calibration["method"])
    ]
    if mismatched_calibration:
        raise ValueError(
            "All default Phase 2C policies must use the configured score calibration method; "
            f"mismatches: {mismatched_calibration}"
        )
    if config["primary_operational_profile"] not in config["operational_profiles"]:
        raise ValueError("primary_operational_profile must exist in operational_profiles.")
    required_weights = {
        "detection_rate",
        "critical_region_recall",
        "missed_engine_rate",
        "false_alarm_engine_rate",
        "healthy_region_false_positive_rate",
        "detected_before_30_fraction",
        "detected_before_60_fraction",
        "alert_instability",
        "utility_variability",
    }
    for profile, weights in config["operational_profiles"].items():
        missing_weights = sorted(required_weights - set(weights))
        if missing_weights:
            raise ValueError(f"Operational profile {profile} missing weights: {missing_weights}")
        for name, value in weights.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"Operational profile weight {profile}.{name} must be non-negative and finite.")
    for key, value in config["feasibility_constraints"].items():
        if float(value) < 0:
            raise ValueError(f"Invalid feasibility constraint {key}.")
    categories = config["health_index_correlation_categories"]
    if not 0 <= float(categories["weak_upper"]) <= float(categories["moderate_upper"]) <= float(categories["strong_lower"]) <= 1:
        raise ValueError("Invalid health-index correlation categories.")
    alert_thresholds = config["operational_alert_thresholds"]
    if not 0 <= float(alert_thresholds["monitor_score"]) <= float(alert_thresholds["warning_score"]) <= float(alert_thresholds["critical_score"]) <= 1:
        raise ValueError("Operational score thresholds must satisfy monitor <= warning <= critical.")
    if not 0 <= float(alert_thresholds["critical_health_index_max"]) <= float(alert_thresholds["warning_health_index_max"]) <= 1:
        raise ValueError("Operational health thresholds must satisfy critical <= warning.")


def load_fd001_frames(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dataset_dir = resolve_project_path(config["dataset_dir"], project_root())
    dataset = load_cmapss_dataset(dataset_dir, str(config["fd001_subset"]))
    train = add_training_rul_targets(dataset.train, rul_cap=float(config["healthy_rul_threshold"]))
    train["true_rul_uncapped"] = train["rul_uncapped"]
    train = add_proxy_labels(
        train,
        healthy_rul_threshold=float(config["healthy_rul_threshold"]),
        critical_rul_threshold=float(config["critical_rul_threshold"]),
    )
    test = derive_test_true_rul_trajectory(dataset.test, dataset.test_rul)
    test = add_proxy_labels(
        test,
        healthy_rul_threshold=float(config["healthy_rul_threshold"]),
        critical_rul_threshold=float(config["critical_rul_threshold"]),
    )
    metadata = {
        "fd001_train_shape": list(dataset.train.shape),
        "fd001_test_shape": list(dataset.test.shape),
        "fd001_train_engine_count": int(dataset.train[UNIT_COLUMN].nunique()),
        "fd001_test_engine_count": int(dataset.test[UNIT_COLUMN].nunique()),
        "fd001_files": {
            "train": str(dataset.files.train),
            "test": str(dataset.files.test),
            "rul": str(dataset.files.rul),
        },
    }
    return train, test, metadata


def load_fd003_test_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_dir = resolve_project_path(config["dataset_dir"], project_root())
    subset = str(config["fd003_subset"]).upper()
    test_path = dataset_dir / f"test_{subset}.txt"
    rul_path = dataset_dir / f"RUL_{subset}.txt"
    test = read_cmapss_table(test_path, test_path.name)
    rul = read_rul_file(rul_path, rul_path.name)
    validate_test_rul_alignment(test, rul)
    test = derive_test_true_rul_trajectory(test, rul)
    test = add_proxy_labels(
        test,
        healthy_rul_threshold=float(config["healthy_rul_threshold"]),
        critical_rul_threshold=float(config["critical_rul_threshold"]),
    )
    return test, {
        "fd003_test_shape": list(test.shape),
        "fd003_test_engine_count": int(test[UNIT_COLUMN].nunique()),
        "fd003_files": {"test": str(test_path), "rul": str(rul_path)},
    }


def phase2_fit_config(config: dict[str, Any], seed: int) -> dict[str, Any]:
    phase2 = load_phase2_config(resolve_project_path(config["phase2_config_path"], project_root()))
    phase2["random_seed"] = int(seed)
    phase2["healthy_rul_threshold"] = float(config["healthy_rul_threshold"])
    phase2["critical_rul_threshold"] = float(config["critical_rul_threshold"])
    return phase2


def fit_apply_fd001_system(
    fitting_frame: pd.DataFrame,
    apply_frames: dict[str, pd.DataFrame],
    phase2_config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    candidates, retained, reasons, feature_audit, correlation_audit = select_anomaly_features(
        fitting_frame,
        include_cycle=bool(phase2_config["include_cycle_as_feature"]),
        configured_exclusions=list(phase2_config["features_to_exclude"]),
        near_constant_threshold=float(phase2_config["near_constant_threshold"]),
        correlation_threshold=float(phase2_config["correlation_threshold"]),
    )
    if not bool(phase2_config["include_cycle_as_feature"]):
        reasons[CYCLE_COLUMN] = "not configured as an anomaly-model feature"
    frames = {"model_train": fitting_frame.copy(), **{name: frame.copy() for name, frame in apply_frames.items()}}
    model_train = frames["model_train"]
    healthy_mask = model_train["true_rul_uncapped"] > float(phase2_config["healthy_rul_threshold"])
    healthy_train = model_train.loc[healthy_mask]
    if healthy_train.empty:
        raise ValueError("No healthy fitting rows are available.")
    scaler = StandardScaler()
    scaler.fit(healthy_train[retained])
    transformed = {split: scaler.transform(frame[retained]) for split, frame in frames.items()}
    healthy_x = scaler.transform(healthy_train[retained])

    health_cfg = dict(phase2_config["health_index"])
    health_model = PCAHealthIndex(
        n_components=health_cfg["n_components"],
        lower_quantile=float(health_cfg["lower_quantile"]),
        upper_quantile=float(health_cfg["upper_quantile"]),
        clip_scaled=bool(health_cfg["clip_scaled"]),
    )
    health_model.fit(transformed["model_train"], model_train["true_rul_uncapped"].to_numpy(dtype=float))
    for split, frame in frames.items():
        raw, scaled = health_model.transform(transformed[split])
        frame["health_index_raw"] = raw
        frame["health_index_scaled"] = scaled
        frames[split] = smooth_by_engine(
            frame,
            value_column="health_index_scaled",
            output_column="smoothed_health_index",
            method=str(phase2_config["smoothing"]["method"]),
            window=int(phase2_config["smoothing"]["window"]),
            causal=bool(phase2_config["smoothing"]["causal"]),
        )

    pca_cfg = dict(phase2_config["pca_reconstruction"])
    pca_detector = PCAReconstructionAnomalyDetector(
        n_components=pca_cfg["n_components"],
        threshold_percentile=float(pca_cfg["threshold_percentile"]),
    ).fit(healthy_x)
    iso_cfg = dict(phase2_config["isolation_forest"])
    iso_detector = IsolationForestAnomalyDetector(**iso_cfg).fit(healthy_x)
    svm_cfg = dict(phase2_config["one_class_svm"])
    svm_detector = OneClassSVMAnomalyDetector(
        kernel=svm_cfg["kernel"],
        nu=float(svm_cfg["nu"]),
        gamma=svm_cfg["gamma"],
        max_training_rows=int(svm_cfg["max_healthy_training_rows"]),
        random_state=int(phase2_config["random_seed"]),
    ).fit(healthy_x)

    for split, frame in frames.items():
        pca_error, pca_score, pca_flag = pca_detector.score(transformed[split])
        iso_score, iso_flag = iso_detector.score(transformed[split])
        svm_score, svm_flag = svm_detector.score(transformed[split])
        frame["pca_reconstruction_error"] = pca_error
        frame["pca_anomaly_score"] = pca_score
        frame["pca_anomaly_flag"] = pca_flag
        frame["isolation_forest_score"] = iso_score
        frame["isolation_forest_flag"] = iso_flag
        frame["one_class_svm_score"] = svm_score
        frame["one_class_svm_flag"] = svm_flag
        frames[split] = frame

    ph_cfg = dict(phase2_config["page_hinkley"])
    for split, frame in frames.items():
        frame, _ = apply_page_hinkley_by_engine(
            frame,
            signal_column=ph_cfg["primary_signal"],
            output_prefix="page_hinkley",
            delta=float(ph_cfg["delta"]),
            threshold=float(ph_cfg["threshold"]),
            min_observations=int(ph_cfg["min_observations"]),
            direction=ph_cfg["direction"],
            reset_after_detection=bool(ph_cfg["reset_after_detection"]),
        )
        frames[split] = frame

    metadata = {
        "candidate_features": candidates,
        "retained_features": retained,
        "excluded_features": reasons,
        "healthy_training_row_count": int(len(healthy_train)),
        "feature_audit_row_count": int(len(feature_audit)),
        "correlation_audit_row_count": int(len(correlation_audit)),
        "scaler_fit_population": "healthy rows from fitting FD001 engines only",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "health_index_explained_variance_ratio": health_model.explained_variance_ratio_,
        "health_index_orientation": health_model.orientation_,
        "pca_reconstruction_threshold": pca_detector.threshold_,
        "pca_reconstruction_explained_variance_ratio": pca_detector.explained_variance_ratio_,
        "isolation_forest_parameters": iso_cfg,
        "one_class_svm_parameters": svm_cfg,
        "one_class_svm_subsampling_applied": svm_detector.subsampling_applied_,
        "one_class_svm_fit_row_count": svm_detector.fit_row_count_,
        "page_hinkley": ph_cfg,
    }
    artifacts = {"scaler": scaler, "retained_features": retained}
    return frames, metadata, artifacts


def calibrate_scores(frames: dict[str, pd.DataFrame], calibration: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "method": calibration["method"],
        "source_population": "healthy rows from fitting FD001 engines only",
        "detectors": {},
    }
    healthy_mask = frames["model_train"]["proxy_degradation_label"] == 0
    for detector, raw_col in RAW_SCORE_COLUMNS.items():
        cal_col = CALIBRATED_SCORE_COLUMNS[detector]
        calibrator = ScoreCalibrator(
            method=str(calibration["method"]),
            lower_quantile=float(calibration["lower_quantile"]),
            upper_quantile=float(calibration["upper_quantile"]),
            epsilon=float(calibration["epsilon"]),
            clip=bool(calibration.get("clip", True)),
        ).fit(frames["model_train"].loc[healthy_mask, raw_col])
        for frame in frames.values():
            frame[cal_col] = calibrator.transform(frame[raw_col])
        metadata["detectors"][detector] = {
            "raw_score_column": raw_col,
            "calibrated_score_column": cal_col,
            **calibrator.metadata(),
        }
    return metadata


def fraction(value: int | float | None, denominator: int | float | None) -> float | None:
    if value is None or denominator in {None, 0}:
        return None
    return float(value) / float(denominator)


def engine_alert_context(result: pd.DataFrame, prefix: str, unstable_threshold: int) -> pd.DataFrame:
    rows = []
    state_col = f"{prefix}_alert_state"
    started_col = f"{prefix}_alert_started"
    flag_col = f"{prefix}_raw_anomaly_flag"
    persistent_col = f"{prefix}_persistent_alarm_state"
    for unit_id, group in result.sort_values([UNIT_COLUMN, CYCLE_COLUMN]).groupby(UNIT_COLUMN):
        states = group[state_col].astype(bool).tolist()
        transitions = sum(1 for left, right in zip(states, states[1:]) if left != right)
        healthy = group[group["proxy_degradation_label"] == 0]
        critical = group[group["proxy_critical_label"] == 1]
        rows.append(
            {
                UNIT_COLUMN: int(unit_id),
                "alert_transition_count": int(transitions),
                "alert_entry_count": int(group[started_col].astype(bool).sum()),
                "alert_active_rows": int(group[state_col].astype(bool).sum()),
                "unstable_alert": bool(transitions > unstable_threshold),
                "multiple_alert_entries": bool(group[started_col].astype(bool).sum() > 1),
                "no_alert": not bool(group[persistent_col].astype(bool).any()),
                "healthy_region_false_positive_rate": None
                if healthy.empty
                else float(healthy[flag_col].astype(bool).mean()),
                "critical_region_recall": None
                if critical.empty
                else float(critical[persistent_col].astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_policy(
    frame: pd.DataFrame,
    policy: dict[str, Any],
    config: dict[str, Any],
    split_name: str,
    output_prefix: str = "locked",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    result, _ = apply_alert_policy(
        frame,
        policy,
        operational_alert_thresholds=config["operational_alert_thresholds"],
        output_prefix=output_prefix,
    )
    score_col = f"{output_prefix}_ensemble_score"
    flag_col = f"{output_prefix}_raw_anomaly_flag"
    persistent_col = f"{output_prefix}_persistent_alarm_state"
    row = row_level_anomaly_metrics(result["proxy_degradation_label"], result[flag_col], result[score_col])
    row["healthy_region_false_positive_rate"] = row["false_positive_rate"]
    critical = result[result["proxy_critical_label"] == 1]
    row["critical_region_recall"] = None if critical.empty else float(critical[persistent_col].astype(bool).mean())
    onset_summary, engine = summarize_engine_onsets(
        result,
        detection_flag_column=persistent_col,
        method_name=str(policy["policy_id"]),
        split_name=split_name,
        healthy_rul_threshold=float(config["healthy_rul_threshold"]),
        critical_rul_threshold=float(config["critical_rul_threshold"]),
    )
    context = engine_alert_context(result, output_prefix, int(config["alert_unstable_transition_threshold"]))
    onset_summary = onset_summary.merge(context, on=UNIT_COLUMN, how="left")
    engines = max(int(engine["engines_evaluated"]), 1)
    detected = int(engine["detected_engines"])
    engine.update(
        {
            "missed_engine_rate": fraction(engine["missed_engines"], engines),
            "false_alarm_engine_rate": fraction(engine["false_alarm_engine_count"], engines),
            "detected_before_60_fraction": fraction(engine["detections_before_60_cycles_rul"], detected),
            "detected_before_30_fraction": fraction(engine["detections_before_30_cycles_rul"], detected),
            "detected_before_critical_fraction": fraction(engine["detections_before_critical_threshold"], detected),
            "median_alert_transitions": None
            if context.empty
            else float(context["alert_transition_count"].median()),
            "mean_alert_transitions": None
            if context.empty
            else float(context["alert_transition_count"].mean()),
            "unstable_engine_fraction": None if context.empty else float(context["unstable_alert"].mean()),
            "no_alert_engine_fraction": None if context.empty else float(context["no_alert"].mean()),
            "multiple_alert_entry_fraction": None if context.empty else float(context["multiple_alert_entries"].mean()),
        }
    )
    transitions = alert_transition_metrics(result, f"{output_prefix}_alert_state", f"{output_prefix}_operational_alert_level")
    return {"row_level": row, "engine_level": engine, "alert_transitions": transitions}, onset_summary, result


def fold_metric_row(
    metrics: dict[str, Any],
    policy: dict[str, Any],
    fold: GroupFold,
    profiles: dict[str, dict[str, float]],
) -> dict[str, Any]:
    row = {
        "policy_id": policy["policy_id"],
        "repeat": fold.repeat,
        "fold": fold.fold,
        "fold_seed": fold.seed,
    }
    row.update(metrics["row_level"])
    engine = metrics["engine_level"]
    row.update(
        {
            "validation_engine_count": engine["engines_evaluated"],
            "detected_engine_count": engine["detected_engines"],
            "engine_detection_rate": engine["detection_rate"],
            "missed_engine_count": engine["missed_engines"],
            "missed_engine_rate": engine["missed_engine_rate"],
            "false_alarm_engine_count": engine["false_alarm_engine_count"],
            "false_alarm_engine_rate": engine["false_alarm_engine_rate"],
            "median_detection_delay": engine["median_detection_delay"],
            "median_lead_time": engine["median_lead_time"],
            "detected_before_60_fraction": engine["detected_before_60_fraction"],
            "detected_before_30_fraction": engine["detected_before_30_fraction"],
            "detected_before_critical_fraction": engine["detected_before_critical_fraction"],
            "median_alert_transitions": engine["median_alert_transitions"],
            "mean_alert_transitions": engine["mean_alert_transitions"],
            "unstable_engine_fraction": engine["unstable_engine_fraction"],
            "no_alert_engine_fraction": engine["no_alert_engine_fraction"],
            "multiple_alert_entry_fraction": engine["multiple_alert_entry_fraction"],
        }
    )
    for profile, weights in profiles.items():
        row[f"utility_{profile}"] = compute_profile_utility(row, weights)
    return row


def run_cross_validation(
    fd001_train: pd.DataFrame,
    policies: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    groups = sorted(fd001_train[UNIT_COLUMN].unique())
    folds = repeated_group_kfold_splits(
        groups,
        n_splits=int(config["group_cross_validation_folds"]),
        n_repeats=int(config["group_cross_validation_repeats"]),
        seeds=list(config["group_cross_validation_seeds"]),
    )
    validate_group_folds(folds, groups, int(config["group_cross_validation_repeats"]))
    rows: list[dict[str, Any]] = []
    split_records = [fold.to_dict() for fold in folds]
    for fold in folds:
        fit = fd001_train[fd001_train[UNIT_COLUMN].isin(fold.train_groups)].copy()
        validation = fd001_train[fd001_train[UNIT_COLUMN].isin(fold.validation_groups)].copy()
        phase2 = phase2_fit_config(config, seed=fold.seed)
        frames, _, _ = fit_apply_fd001_system(fit, {"validation": validation}, phase2)
        calibrate_scores(frames, config["score_calibration"])
        validation_frame = frames["validation"]
        for policy in policies:
            metrics, _, _ = evaluate_policy(validation_frame, policy, config, split_name="fd001_cv", output_prefix="cv")
            rows.append(fold_metric_row(metrics, policy, fold, config["operational_profiles"]))
    return pd.DataFrame(rows), split_records


def add_bootstrap_intervals(
    fd001_summary: pd.DataFrame,
    fd003_summary: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    funcs = alert_engine_metric_functions()
    funcs.update(
        {
            "healthy_region_false_positive_rate": lambda frame: None
            if pd.to_numeric(frame["healthy_region_false_positive_rate"], errors="coerce").dropna().empty
            else float(pd.to_numeric(frame["healthy_region_false_positive_rate"], errors="coerce").dropna().mean()),
            "critical_region_recall": lambda frame: None
            if pd.to_numeric(frame["critical_region_recall"], errors="coerce").dropna().empty
            else float(pd.to_numeric(frame["critical_region_recall"], errors="coerce").dropna().mean()),
        }
    )
    samples = int(config["bootstrap_samples"])
    confidence = float(config["confidence_level"])
    seed = int(config["bootstrap_seed"])
    return {
        "fd001_development_test": bootstrap_engine_metrics(fd001_summary, funcs, samples, confidence, seed),
        "fd003_external": bootstrap_engine_metrics(fd003_summary, funcs, samples, confidence, seed + 1),
        "method": "deterministic engine-level bootstrap; engines resampled with replacement",
    }


def scaler_outside_fraction(frame: pd.DataFrame, scaler: StandardScaler, features: list[str], limit: float = 3.0) -> float:
    transformed = scaler.transform(frame[features])
    return float((np.abs(transformed) > limit).mean())


def health_transfer_table(frames: dict[str, pd.DataFrame], categories: dict[str, float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, frame in frames.items():
        for unit_id, group in frame.sort_values([UNIT_COLUMN, CYCLE_COLUMN]).groupby(UNIT_COLUMN):
            corr = group[["smoothed_health_index", "true_rul_uncapped"]].corr(method="spearman").iloc[0, 1]
            if pd.isna(corr):
                label = "undefined"
            elif corr < 0:
                label = "negative"
            elif corr < float(categories["weak_upper"]):
                label = "weak"
            elif corr < float(categories["moderate_upper"]):
                label = "moderate"
            else:
                label = "strong"
            rows.append(
                {
                    "split": split,
                    "unit_id": int(unit_id),
                    "trajectory_length": int(len(group)),
                    "initial_true_rul": float(group["true_rul_uncapped"].iloc[0]),
                    "final_true_rul": float(group["true_rul_uncapped"].iloc[-1]),
                    "initial_health_index": float(group["smoothed_health_index"].iloc[0]),
                    "final_health_index": float(group["smoothed_health_index"].iloc[-1]),
                    "spearman_correlation": None if pd.isna(corr) else float(corr),
                    "correlation_category": label,
                    "correct_overall_direction": bool(False if pd.isna(corr) else corr >= 0),
                }
            )
    table = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for split, group in table.groupby("split"):
        correlations = pd.to_numeric(group["spearman_correlation"], errors="coerce").dropna()
        summary[split] = {
            "aggregate_spearman": float(frames[split][["smoothed_health_index", "true_rul_uncapped"]].corr(method="spearman").iloc[0, 1]),
            "median_per_engine_spearman": None if correlations.empty else float(correlations.median()),
            "iqr_per_engine_spearman": None
            if correlations.empty
            else float(correlations.quantile(0.75) - correlations.quantile(0.25)),
            "category_counts": group["correlation_category"].value_counts().to_dict(),
            "correct_overall_direction_fraction": float(group["correct_overall_direction"].mean()),
            "initial_health_index_median": float(group["initial_health_index"].median()),
            "final_health_index_median": float(group["final_health_index"].median()),
            "trajectory_length_correlation": None
            if correlations.empty
            else float(group[["trajectory_length", "spearman_correlation"]].corr(method="spearman").iloc[0, 1]),
            "initial_rul_correlation": None
            if correlations.empty
            else float(group[["initial_true_rul", "spearman_correlation"]].corr(method="spearman").iloc[0, 1]),
        }
    return table, summary


def make_domain_shift(
    reference_frame: pd.DataFrame,
    fd001_frame: pd.DataFrame,
    fd003_frame: pd.DataFrame,
    artifacts: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = list(artifacts["retained_features"])
    healthy_ref = reference_frame[reference_frame["proxy_degradation_label"] == 0]
    table = feature_shift_table(healthy_ref, fd001_frame, fd003_frame, features)
    score_cols = [
        "pca_anomaly_score",
        "isolation_forest_score",
        "one_class_svm_score",
        *CALIBRATED_SCORE_COLUMNS.values(),
        "smoothed_health_index",
    ]
    summary: dict[str, Any] = {
        "reference_population": "healthy rows from all FD001 training engines",
        "fd001_development_test": {
            **trajectory_summary(fd001_frame, "fd001_development_test"),
            **distribution_summary(fd001_frame, score_cols, "fd001_development_test"),
            "locked_scaler_abs_gt_3_fraction": scaler_outside_fraction(fd001_frame, artifacts["scaler"], features),
        },
        "fd003_external": {
            **trajectory_summary(fd003_frame, "fd003_external"),
            **distribution_summary(fd003_frame, score_cols, "fd003_external"),
            "locked_scaler_abs_gt_3_fraction": scaler_outside_fraction(fd003_frame, artifacts["scaler"], features),
        },
        "largest_fd003_feature_smd": None,
        "largest_fd003_feature_psi": None,
        "diagnostic_only": True,
    }
    if not table.empty:
        smd = table.dropna(subset=["fd003_standardized_mean_difference"]).copy()
        if not smd.empty:
            smd["abs_smd"] = smd["fd003_standardized_mean_difference"].abs()
            top = smd.sort_values("abs_smd", ascending=False).iloc[0]
            summary["largest_fd003_feature_smd"] = {
                "feature": top["feature"],
                "value": float(top["fd003_standardized_mean_difference"]),
            }
        psi = table.dropna(subset=["fd003_psi"]).copy()
        if not psi.empty:
            top = psi.sort_values("fd003_psi", ascending=False).iloc[0]
            summary["largest_fd003_feature_psi"] = {
                "feature": top["feature"],
                "value": float(top["fd003_psi"]),
            }
    table.to_csv(output_dir / "domain_shift_features.csv", index=False)
    write_json(output_dir / "domain_shift_summary.json", summary)
    return table, summary


def make_figures(
    output_dir: Path,
    fold_metrics: pd.DataFrame,
    policy_summary: pd.DataFrame,
    policy_ranking: pd.DataFrame,
    fd001_metrics: dict[str, Any],
    fd003_metrics: dict[str, Any],
    bootstrap: dict[str, Any],
    domain_features: pd.DataFrame,
    health_transfer: pd.DataFrame,
    fd001_frame: pd.DataFrame,
    fd003_frame: pd.DataFrame,
    locked_policy: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    figures: list[str] = []
    figures_dir = output_dir / "figures"
    timelines_dir = output_dir / "engine_timelines"
    figures_dir.mkdir(parents=True, exist_ok=True)
    timelines_dir.mkdir(parents=True, exist_ok=True)

    def save(path: Path) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        figures.append(str(path))

    plt.figure(figsize=(10, 5))
    fold_metrics.boxplot(column="engine_detection_rate", by="policy_id", rot=90)
    plt.suptitle("")
    plt.title("Cross-Validation Detection-Rate Distribution")
    plt.ylabel("Detection rate")
    save(figures_dir / "cv_detection_rate_distribution_by_policy.png")

    plt.figure(figsize=(10, 5))
    fold_metrics.boxplot(column="false_alarm_engine_rate", by="policy_id", rot=90)
    plt.suptitle("")
    plt.title("Cross-Validation False-Alarm-Rate Distribution")
    plt.ylabel("False-alarm engine rate")
    save(figures_dir / "cv_false_alarm_rate_distribution_by_policy.png")

    profile = str(config["primary_operational_profile"])
    plt.figure(figsize=(10, 5))
    ordered = policy_ranking.sort_values(f"robust_utility_{profile}", ascending=True)
    plt.barh(ordered["policy_id"], ordered[f"robust_utility_{profile}"])
    plt.title("Cross-Validation Robust-Utility Ranking")
    save(figures_dir / "cv_robust_utility_ranking.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(policy_summary[f"mean_utility_{profile}"], policy_summary[f"std_utility_{profile}"])
    for _, row in policy_summary.iterrows():
        plt.annotate(row["policy_id"], (row[f"mean_utility_{profile}"], row[f"std_utility_{profile}"]), fontsize=6)
    plt.xlabel("Mean utility")
    plt.ylabel("Utility standard deviation")
    plt.title("Mean Utility Versus Utility Variability")
    save(figures_dir / "mean_utility_vs_variability.png")

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        policy_summary["false_alarm_engine_rate_mean"],
        policy_summary["engine_detection_rate_mean"],
        xerr=policy_summary["false_alarm_engine_rate_std"],
        yerr=policy_summary["engine_detection_rate_std"],
        fmt="o",
        alpha=0.75,
    )
    plt.xlabel("False-alarm engine rate")
    plt.ylabel("Detection rate")
    plt.title("Detection Rate Versus False-Alarm Rate")
    save(figures_dir / "detection_rate_vs_false_alarm_rate_errorbars.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(policy_summary["missed_engine_rate_mean"], policy_summary["critical_region_recall_mean"])
    plt.xlabel("Missed-engine rate")
    plt.ylabel("Critical-region recall")
    plt.title("Missed Engines Versus Critical Recall")
    save(figures_dir / "missed_engine_rate_vs_critical_recall.png")

    labels = ["FD001 dev-test", "FD003 external"]
    plt.figure(figsize=(8, 5))
    x = np.arange(len(labels))
    plt.bar(x - 0.15, [fd001_metrics["engine_level"]["detection_rate"], fd003_metrics["engine_level"]["detection_rate"]], width=0.3, label="detection")
    plt.bar(x + 0.15, [fd001_metrics["engine_level"]["false_alarm_engine_rate"], fd003_metrics["engine_level"]["false_alarm_engine_rate"]], width=0.3, label="false alarm")
    plt.xticks(x, labels)
    plt.legend()
    plt.title("FD001 Versus FD003 Metric Comparison")
    save(figures_dir / "fd001_vs_fd003_metric_comparison.png")

    plt.figure(figsize=(8, 5))
    ci_names = ["detection_rate", "false_alarm_engine_rate", "median_lead_time"]
    for idx, dataset in enumerate(["fd001_development_test", "fd003_external"]):
        lows = [bootstrap[dataset][name]["ci_lower"] for name in ci_names]
        highs = [bootstrap[dataset][name]["ci_upper"] for name in ci_names]
        estimates = [bootstrap[dataset][name]["estimate"] for name in ci_names]
        y = np.arange(len(ci_names)) + idx * 0.2
        xerr = [
            [0 if low is None or est is None else est - low for low, est in zip(lows, estimates)],
            [0 if high is None or est is None else high - est for high, est in zip(highs, estimates)],
        ]
        plt.errorbar(estimates, y, xerr=xerr, fmt="o", label=dataset)
    plt.yticks(np.arange(len(ci_names)) + 0.1, ci_names)
    plt.legend()
    plt.title("Bootstrap Confidence-Interval Comparison")
    save(figures_dir / "bootstrap_confidence_interval_comparison.png")

    plt.figure(figsize=(8, 5))
    top_shift = domain_features.assign(abs_smd=domain_features["fd003_standardized_mean_difference"].abs()).sort_values("abs_smd", ascending=False).head(12)
    plt.barh(top_shift["feature"], top_shift["fd003_standardized_mean_difference"])
    plt.title("Feature Domain-Shift Overview")
    save(figures_dir / "feature_domain_shift_overview.png")

    plt.figure(figsize=(8, 5))
    for label, frame in [("FD001", fd001_frame), ("FD003", fd003_frame)]:
        plt.hist(frame["locked_ensemble_score"], bins=30, alpha=0.5, label=label)
    plt.legend()
    plt.title("Detector-Score Distribution Shift")
    save(figures_dir / "detector_score_distribution_shift.png")

    plt.figure(figsize=(8, 5))
    for label, frame in [("FD001", fd001_frame), ("FD003", fd003_frame)]:
        plt.hist(frame["smoothed_health_index"], bins=30, alpha=0.5, label=label)
    plt.legend()
    plt.title("Health-Index Transfer Comparison")
    save(figures_dir / "health_index_transfer_comparison.png")

    plt.figure(figsize=(8, 5))
    health_transfer.boxplot(column="spearman_correlation", by="split")
    plt.suptitle("")
    plt.title("Per-Engine Health-Index Correlation Comparison")
    save(figures_dir / "per_engine_health_correlation_comparison.png")

    plt.figure(figsize=(8, 5))
    lengths = []
    names = []
    for label, frame in [("FD001", fd001_frame), ("FD003", fd003_frame)]:
        for _, group in frame.groupby(UNIT_COLUMN):
            lengths.append(len(group))
            names.append(label)
    pd.DataFrame({"split": names, "trajectory_length": lengths}).boxplot(column="trajectory_length", by="split")
    plt.suptitle("")
    plt.title("Trajectory-Length Comparison")
    save(figures_dir / "trajectory_length_comparison.png")

    def timeline_units(frame: pd.DataFrame) -> list[int]:
        chosen: list[int] = []
        summaries = []
        for unit_id, group in frame.groupby(UNIT_COLUMN):
            alarms = group[group["locked_persistent_alarm_state"].astype(bool)]
            if alarms.empty:
                category = "missed"
            else:
                first_rul = float(alarms["true_rul_uncapped"].iloc[0])
                if first_rul > float(config["healthy_rul_threshold"]):
                    category = "false_alarm"
                elif first_rul <= float(config["critical_rul_threshold"]):
                    category = "late"
                else:
                    category = "early"
            summaries.append((int(unit_id), category))
        for category in ["early", "late", "missed", "false_alarm"]:
            match = [unit for unit, item in summaries if item == category and unit not in chosen]
            if match:
                chosen.append(match[0])
            if len(chosen) >= int(config["representative_timeline_count"]):
                break
        for unit, _ in summaries:
            if len(chosen) >= int(config["representative_timeline_count"]):
                break
            if unit not in chosen:
                chosen.append(unit)
        return chosen

    def write_timelines(frame: pd.DataFrame, label: str) -> None:
        for unit_id in timeline_units(frame):
            group = frame[frame[UNIT_COLUMN] == unit_id].sort_values(CYCLE_COLUMN)
            fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
            axes[0].plot(group[CYCLE_COLUMN], group["locked_ensemble_score"], label="locked score")
            axes[0].axhline(float(locked_policy["threshold"]), linestyle="--", color="red", label="policy threshold")
            axes[0].legend(fontsize=8)
            axes[1].plot(group[CYCLE_COLUMN], group["smoothed_health_index"], color="tab:green", label="health")
            axes[1].legend(fontsize=8)
            axes[2].step(group[CYCLE_COLUMN], group["locked_persistent_alarm_state"].astype(int), where="post", label="persistent")
            axes[2].step(group[CYCLE_COLUMN], group["locked_alert_state"].astype(int), where="post", label="hysteresis")
            axes[2].legend(fontsize=8)
            axes[3].plot(group[CYCLE_COLUMN], group["true_rul_uncapped"], label="true RUL")
            onset = group.loc[group["proxy_degradation_label"] == 1, CYCLE_COLUMN]
            critical = group.loc[group["proxy_critical_label"] == 1, CYCLE_COLUMN]
            if len(onset):
                axes[3].axvline(onset.iloc[0], linestyle="--", color="orange", label="proxy onset")
            if len(critical):
                axes[3].axvline(critical.iloc[0], linestyle="--", color="red", label="critical")
            axes[3].legend(fontsize=8)
            axes[3].set_xlabel("Cycle")
            fig.suptitle(f"{label} Engine {unit_id} Locked Alert Timeline")
            path = timelines_dir / f"{label.lower()}_engine_{int(unit_id):03d}_locked_timeline.png"
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close(fig)
            figures.append(str(path))

    write_timelines(fd001_frame, "FD001")
    write_timelines(fd003_frame, "FD003")
    return figures


def write_design_note(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# FD001 Alert Generalization Design

Phase 2C tests whether the Phase 2B-style alert policy generalizes beyond one 20-engine validation split. The prior validation-to-test gap makes single-split selection risky, especially after evaluating many operating points on a small validation group. The phase locks one bounded, transparent policy using FD001 training-engine cross-validation only.

## Evaluation Integrity

FD001 test is labelled as previously observed development-test evaluation because earlier phases already exposed it. It is not used for policy selection or tuning. FD003 is the first untouched external transfer evaluation. FD003 results do not change feature selection, scaling, PCA orientation, detector fitting, score calibration, threshold, persistence, hysteresis, policy choice, or conclusions.

## Cross-Validation And Policy Registry

Repeated group cross-validation uses complete FD001 training engines. Rows from one engine never appear in both fitting and validation portions of a fold. Each fold refits constant-feature checks, retained features, healthy-row scaler, PCA health index, PCA reconstruction, Isolation Forest, One-Class SVM, and score-calibration statistics using only fold-fitting engines.

Candidate policies are fixed in configuration before cross-validation. The registry is bounded, transparent, and includes the Phase 2B selected policy plus maximum-score, mean, median, Isolation/SVM weighted mean, two-of-three voting, Isolation-only, SVM-only, and PCA-only references. The registry validator rejects duplicate IDs, invalid weights, invalid thresholds, invalid persistence, invalid hysteresis, missing profiles, and excessive candidate counts.

## Variance-Aware Selection

Policy summaries report cross-fold mean, standard deviation, median, min, max, and 5th/95th percentiles. Selection prefers feasible policies, ranks by robust utility, and uses deterministic tie-breakers. Robust utility is mean utility minus the configured variability penalty times utility standard deviation.

## External Transfer, Bootstrap, And Diagnostics

After locking, the final system is fit once on all FD001 training engines, then applied unchanged to FD001 development test and FD003 external test. Bootstrap confidence intervals resample engines, not rows. Domain-shift and health-index transfer analyses are diagnostic only and do not retune the policy.

## Limitations And Originality

Proxy labels are RUL-threshold evaluation proxies, not certified physical anomaly truth. FD001 test is exposed from prior phases. FD003 is direct transfer only. PSI and domain-shift metrics are descriptive. This implementation is original AeroGuard code using existing classical components, no deep learning, no external services, no package changes, and no unbounded searches.
""",
        encoding="utf-8",
        newline="\n",
    )


def write_results_note(path: Path, result: dict[str, Any], config_path: Path) -> None:
    locked = result["locked_policy"]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# FD001 Alert Generalization Results\n\n")
        handle.write("FD001 test is a previously observed development-test benchmark. FD003 is the untouched external transfer evaluation.\n\n")
        handle.write(f"- Python interpreter: `{result['python_executable']}`\n")
        handle.write(f"- Python version: `{result['python_version']}`\n")
        handle.write(f"- Dataset files: `{result['dataset_files']}`\n")
        handle.write(f"- Dataset dimensions: `{result['dataset_dimensions']}`\n")
        handle.write(f"- Cross-validation folds/repeats: `{result['cv_folds']}` / `{result['cv_repeats']}`\n")
        handle.write(f"- Candidate policies: `{result['candidate_policy_count']}`\n")
        handle.write(f"- Locked policy: `{locked['policy_id']}`\n")
        handle.write(f"- Selection rationale: `{locked['selection_rationale']}`\n")
        handle.write(f"- Runtime seconds: `{result['runtime_seconds']:.3f}`\n\n")
        handle.write("## Cross-Validation Metrics\n\n")
        handle.write(f"`{result['locked_policy_cv_metrics']}`\n\n")
        handle.write("## FD001 Development-Test Metrics\n\n")
        handle.write(f"`{result['fd001_development_test_metrics']}`\n\n")
        handle.write("## FD003 External Metrics\n\n")
        handle.write(f"`{result['fd003_external_metrics']}`\n\n")
        handle.write("## Bootstrap Confidence Intervals\n\n")
        handle.write(f"`{result['bootstrap_confidence_intervals']}`\n\n")
        handle.write("## Domain Shift Findings\n\n")
        handle.write(f"`{result['domain_shift_findings']}`\n\n")
        handle.write("## Health-Index Transfer Findings\n\n")
        handle.write(f"`{result['health_index_transfer_findings']}`\n\n")
        handle.write("## Generalization Classification\n\n")
        handle.write(f"`{result['generalization_conclusion']}`\n\n")
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
            "python -m aeroguard.pipelines.validate_fd001_alert_generalization "
            f'--config "{config_path.as_posix()}"\n'
        )
        handle.write("```\n")


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    start = time.perf_counter()
    config_path = Path(config_path)
    root = project_root()
    config = load_config(config_path)
    output_dir = resolve_project_path(config["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "engine_timelines").mkdir(parents=True, exist_ok=True)

    policies = validate_policy_registry(
        list(config["candidate_policy_registry"]),
        int(config["maximum_policy_count"]),
        dict(config["operational_profiles"]),
    )
    write_json(output_dir / "candidate_policy_registry.json", {"policies": policies})

    fd001_train, fd001_test, fd001_meta = load_fd001_frames(config)
    fd003_test, fd003_meta = load_fd003_test_frame(config)
    fold_metrics, split_records = run_cross_validation(fd001_train, policies, config)
    write_json(output_dir / "cross_validation_splits.json", {"folds": split_records})
    fold_metrics.to_csv(output_dir / "cross_validation_fold_metrics.csv", index=False)
    policy_summary, policy_ranking = summarize_policy_folds(
        fold_metrics,
        config["operational_profiles"],
        float(config["utility_variability_penalty"]),
        config["feasibility_constraints"],
    )
    policy_summary.to_csv(output_dir / "cross_validation_policy_summary.csv", index=False)
    policy_ranking.to_csv(output_dir / "cross_validation_policy_ranking.csv", index=False)
    selected_summary = select_locked_policy(policy_summary, str(config["primary_operational_profile"]))
    locked_policy = next(policy for policy in policies if policy["policy_id"] == selected_summary["policy_id"])

    phase2 = phase2_fit_config(config, seed=int(config["random_seed"]))
    final_frames, fit_metadata, artifacts = fit_apply_fd001_system(
        fd001_train,
        {"fd001_development_test": fd001_test, "fd003_external": fd003_test},
        phase2,
    )
    calibration_metadata = calibrate_scores(final_frames, config["score_calibration"])
    fd001_metrics, fd001_engine_summary, fd001_alerts = evaluate_policy(
        final_frames["fd001_development_test"],
        locked_policy,
        config,
        split_name="fd001_previously_observed_development_test",
    )
    fd003_metrics, fd003_engine_summary, fd003_alerts = evaluate_policy(
        final_frames["fd003_external"],
        locked_policy,
        config,
        split_name="fd003_untouched_external_transfer",
    )
    fd001_metrics["engine_level"]["false_alarm_engine_rate"] = fraction(
        fd001_metrics["engine_level"]["false_alarm_engine_count"],
        fd001_metrics["engine_level"]["engines_evaluated"],
    )
    fd003_metrics["engine_level"]["false_alarm_engine_rate"] = fraction(
        fd003_metrics["engine_level"]["false_alarm_engine_count"],
        fd003_metrics["engine_level"]["engines_evaluated"],
    )
    fd001_engine_summary.to_csv(output_dir / "fd001_development_test_engine_summary.csv", index=False)
    fd003_engine_summary.to_csv(output_dir / "fd003_external_engine_summary.csv", index=False)
    fd003_cycle_cols = [
        UNIT_COLUMN,
        CYCLE_COLUMN,
        "true_rul_uncapped",
        "proxy_degradation_label",
        "proxy_critical_label",
        *RAW_SCORE_COLUMNS.values(),
        *CALIBRATED_SCORE_COLUMNS.values(),
        "locked_ensemble_score",
        "locked_raw_anomaly_flag",
        "locked_persistent_alarm_state",
        "locked_alert_state",
        "locked_operational_alert_level",
        "smoothed_health_index",
        "page_hinkley_change_flag",
    ]
    fd003_alerts[[col for col in fd003_cycle_cols if col in fd003_alerts.columns]].to_csv(
        output_dir / "fd003_cycle_level_alerts.csv",
        index=False,
    )

    fd001_metrics["label"] = "FD001 previously observed development-test evaluation"
    fd003_metrics["label"] = "FD003 untouched external transfer evaluation"
    write_json(output_dir / "fd001_development_test_metrics.json", fd001_metrics)
    write_json(output_dir / "fd003_external_metrics.json", fd003_metrics)
    fit_metadata.update(
        {
            "fitting_population": "all FD001 training engines only",
            "score_calibration": calibration_metadata,
            "locked_policy_not_selected_from_test": True,
        }
    )
    write_json(output_dir / "final_fd001_fit_metadata.json", fit_metadata)

    bootstrap = add_bootstrap_intervals(fd001_engine_summary, fd003_engine_summary, config)
    write_json(output_dir / "bootstrap_confidence_intervals.json", bootstrap)
    domain_features, domain_summary = make_domain_shift(
        final_frames["model_train"],
        fd001_alerts,
        fd003_alerts,
        artifacts,
        output_dir,
    )
    health_table, health_summary = health_transfer_table(
        {"fd001_development_test": fd001_alerts, "fd003_external": fd003_alerts},
        config["health_index_correlation_categories"],
    )
    health_table.to_csv(output_dir / "health_index_transfer.csv", index=False)
    write_json(output_dir / "health_index_transfer_summary.json", health_summary)

    conclusion = classify_generalization(
        config["generalization_classification_criteria"],
        selected_summary,
        fd001_metrics,
        fd003_metrics,
        bootstrap,
    )
    write_json(output_dir / "generalization_conclusion.json", conclusion)

    primary_profile = str(config["primary_operational_profile"])
    locked_payload = {
        "policy_id": locked_policy["policy_id"],
        "selection_profile": primary_profile,
        "policy": locked_policy,
        "mean_cross_validation_metrics": {
            key: value for key, value in selected_summary.items() if key.endswith("_mean")
        },
        "standard_deviations": {
            key: value for key, value in selected_summary.items() if key.endswith("_std")
        },
        "robust_utility": selected_summary[f"robust_utility_{primary_profile}"],
        "mean_utility": selected_summary[f"mean_utility_{primary_profile}"],
        "feasible": selected_summary["feasible"],
        "failed_constraints": selected_summary["failed_constraints"],
        "selection_rationale": (
            f"Selected by {primary_profile} robust utility using repeated FD001 training-engine "
            "cross-validation only; FD001 development-test and FD003 external results were not used."
        ),
        "reproduction": {
            "pipeline": "aeroguard.pipelines.validate_fd001_alert_generalization",
            "config": config_path.as_posix(),
        },
        "test_results_used_for_selection": False,
    }
    write_json(output_dir / "locked_policy.json", locked_payload)

    figures = make_figures(
        output_dir,
        fold_metrics,
        policy_summary,
        policy_ranking,
        fd001_metrics,
        fd003_metrics,
        bootstrap,
        domain_features,
        health_table,
        fd001_alerts,
        fd003_alerts,
        locked_policy,
        config,
    )

    design_note = root / "notes" / "fd001_alert_generalization_design.md"
    results_note = root / "notes" / "fd001_alert_generalization_results.md"
    run_summary_path = output_dir / "run_summary.json"
    generated_files = [
        str(output_dir / "cross_validation_splits.json"),
        str(output_dir / "candidate_policy_registry.json"),
        str(output_dir / "cross_validation_fold_metrics.csv"),
        str(output_dir / "cross_validation_policy_summary.csv"),
        str(output_dir / "cross_validation_policy_ranking.csv"),
        str(output_dir / "locked_policy.json"),
        str(output_dir / "final_fd001_fit_metadata.json"),
        str(output_dir / "fd001_development_test_metrics.json"),
        str(output_dir / "fd001_development_test_engine_summary.csv"),
        str(output_dir / "fd003_external_metrics.json"),
        str(output_dir / "fd003_external_engine_summary.csv"),
        str(output_dir / "fd003_cycle_level_alerts.csv"),
        str(output_dir / "bootstrap_confidence_intervals.json"),
        str(output_dir / "domain_shift_features.csv"),
        str(output_dir / "domain_shift_summary.json"),
        str(output_dir / "health_index_transfer.csv"),
        str(output_dir / "health_index_transfer_summary.json"),
        str(output_dir / "generalization_conclusion.json"),
        *figures,
        str(design_note),
        str(results_note),
        str(run_summary_path),
    ]
    runtime = time.perf_counter() - start
    result = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "dataset_files": {
            "fd001": fd001_meta["fd001_files"],
            "fd003": fd003_meta["fd003_files"],
        },
        "dataset_dimensions": {
            "fd001_train_shape": fd001_meta["fd001_train_shape"],
            "fd001_test_shape": fd001_meta["fd001_test_shape"],
            "fd003_test_shape": fd003_meta["fd003_test_shape"],
            "fd001_train_engine_count": fd001_meta["fd001_train_engine_count"],
            "fd001_test_engine_count": fd001_meta["fd001_test_engine_count"],
            "fd003_test_engine_count": fd003_meta["fd003_test_engine_count"],
        },
        "cv_folds": int(config["group_cross_validation_folds"]),
        "cv_repeats": int(config["group_cross_validation_repeats"]),
        "candidate_policy_count": int(len(policies)),
        "locked_policy": locked_payload,
        "locked_policy_cv_metrics": selected_summary,
        "fd001_development_test_metrics": fd001_metrics,
        "fd003_external_metrics": fd003_metrics,
        "bootstrap_confidence_intervals": bootstrap,
        "domain_shift_findings": domain_summary,
        "health_index_transfer_findings": health_summary,
        "generalization_conclusion": conclusion,
        "runtime_seconds": runtime,
        "generated_files": generated_files,
        "warnings": [
            "FD001 test was previously observed in earlier phases and is not described as untouched.",
            "FD003 was evaluated only after locking the policy and was not used for tuning.",
            "Proxy labels are RUL-threshold evaluation proxies, not certified physical anomaly truth.",
            "Domain-shift and health-transfer analyses are diagnostic only.",
        ],
    }
    write_design_note(design_note)
    write_results_note(results_note, result, config_path)
    write_json(run_summary_path, result)
    print(json.dumps(_json_ready(result), indent=2, allow_nan=False))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate FD001 alert-policy generalization and FD003 transfer.")
    parser.add_argument("--config", required=True, help="Path to Phase 2C YAML configuration.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
