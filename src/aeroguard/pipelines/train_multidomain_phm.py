"""Multidomain classical PHM training and FD004 transfer evaluation."""

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
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from aeroguard.anomaly.isolation_forest import IsolationForestAnomalyDetector
from aeroguard.anomaly.one_class_svm import OneClassSVMAnomalyDetector
from aeroguard.anomaly.pca_reconstruction import PCAReconstructionAnomalyDetector
from aeroguard.anomaly.policy_registry import CALIBRATED_SCORE_COLUMNS, apply_alert_policy, validate_policy_registry
from aeroguard.anomaly.score_calibration import ScoreCalibrator
from aeroguard.data.columns import BASE_FEATURE_COLUMNS, CYCLE_COLUMN, OPERATIONAL_SETTING_COLUMNS, SENSOR_COLUMNS, UNIT_COLUMN
from aeroguard.data.multi_subset import load_test_subset, load_test_subsets, load_training_subsets
from aeroguard.data.operating_regimes import OperatingRegimeModel
from aeroguard.evaluation.alert_metrics import alert_transition_metrics
from aeroguard.evaluation.anomaly_metrics import row_level_anomaly_metrics, summarize_engine_onsets
from aeroguard.evaluation.bootstrap import bootstrap_engine_metrics
from aeroguard.evaluation.leave_one_domain_out import (
    DomainSplit,
    leave_one_domain_out_splits,
    stratified_engine_group_splits,
    validate_no_engine_leakage,
)
from aeroguard.evaluation.metrics import regression_metrics
from aeroguard.evaluation.transfer_metrics import classify_transfer, method_utility, summarize_method_metrics
from aeroguard.features.condition_normalization import ConditionNormalizer
from aeroguard.features.domain_features import domain_feature_audit
from aeroguard.health.pca_health_index import PCAHealthIndex
from aeroguard.health.smoothing import smooth_by_engine
from aeroguard.pipelines.train_fd001_baseline import project_root, resolve_project_path
from aeroguard.pipelines.train_fd001_health_anomaly import select_anomaly_features


RAW_SCORE_COLUMNS = {
    "pca_reconstruction": "pca_anomaly_score",
    "isolation_forest": "isolation_forest_score",
    "one_class_svm": "one_class_svm_score",
}

REQUIRED_CONFIG_KEYS = {
    "dataset_dir",
    "training_subsets",
    "development_test_subsets",
    "external_subset",
    "random_seed",
    "healthy_rul_threshold",
    "critical_rul_threshold",
    "validation",
    "candidate_normalization_methods",
    "operating_regime_counts",
    "residualization",
    "method_registry",
    "maximum_method_count",
    "utility_weights",
    "feasibility_constraints",
    "detectors",
    "health_index",
    "smoothing",
    "score_calibration",
    "operational_alert_thresholds",
    "rul_baseline",
    "bootstrap_samples",
    "confidence_level",
    "bootstrap_seed",
    "generalization_criteria",
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
    if isinstance(value, (np.bool_,)):
        return bool(value)
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
    dataset_dir = resolve_project_path(config["dataset_dir"], root)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    training = [str(item).upper() for item in config["training_subsets"]]
    development = [str(item).upper() for item in config["development_test_subsets"]]
    external = str(config["external_subset"]).upper()
    if external in training:
        raise ValueError("external_subset must not be part of training_subsets.")
    for subset in training:
        if not (dataset_dir / f"train_{subset}.txt").exists():
            raise FileNotFoundError(f"Missing training file for {subset}.")
    for subset in sorted(set(training + development + [external])):
        for filename in [f"test_{subset}.txt", f"RUL_{subset}.txt"]:
            if not (dataset_dir / filename).exists():
                raise FileNotFoundError(f"Missing test/RUL file: {dataset_dir / filename}")
    if float(config["healthy_rul_threshold"]) <= float(config["critical_rul_threshold"]):
        raise ValueError("healthy_rul_threshold must be greater than critical_rul_threshold.")
    validation = config["validation"]
    if int(validation["folds"]) < 2 or int(validation["repeats"]) < 1:
        raise ValueError("Invalid validation fold/repeat settings.")
    if len(validation["seeds"]) != int(validation["repeats"]):
        raise ValueError("validation.seeds length must equal validation.repeats.")
    if int(config["maximum_method_count"]) <= 0:
        raise ValueError("maximum_method_count must be positive.")
    if len(config["method_registry"]) > int(config["maximum_method_count"]):
        raise ValueError("method_registry exceeds maximum_method_count.")
    if int(config["bootstrap_samples"]) <= 0 or not 0 < float(config["confidence_level"]) < 1:
        raise ValueError("Invalid bootstrap configuration.")
    output_dir = resolve_project_path(config["output_dir"], root)
    checked_paths = [output_dir]
    if "design_note_path" in config:
        checked_paths.append(resolve_project_path(config["design_note_path"], root))
    if "results_note_path" in config:
        checked_paths.append(resolve_project_path(config["results_note_path"], root))
    for path in checked_paths:
        lowered = str(path).lower()
        if "\\references\\" in lowered or "\\extracted-code\\" in lowered:
            raise ValueError("Output paths must not be inside protected folders.")
    for method in config["method_registry"]:
        if method["normalization_method"] not in config["candidate_normalization_methods"]:
            raise ValueError(f"Method {method['method_id']} references unsupported normalization method.")
    validate_policy_registry(
        [method["policy"] for method in config["method_registry"]],
        maximum_policy_count=int(config["maximum_method_count"]),
        operational_profiles={"balanced": config["utility_weights"]},
    )


def assign_working_unit_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Use collision-safe numeric unit IDs for existing engine-wise helpers."""
    result = frame.copy()
    mapping = {engine: idx + 1 for idx, engine in enumerate(sorted(result["global_engine_id"].unique()))}
    result[UNIT_COLUMN] = result["global_engine_id"].map(mapping).astype(int)
    return result


def method_policy(method: dict[str, Any]) -> dict[str, Any]:
    return dict(method["policy"])


def prepare_method_frames(
    fitting_frame: pd.DataFrame,
    apply_frames: dict[str, pd.DataFrame],
    method: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    candidates, retained, reasons, _, _ = select_anomaly_features(
        fitting_frame,
        include_cycle=bool(config.get("include_cycle_as_feature", False)),
        configured_exclusions=list(config.get("features_to_exclude", [])),
        near_constant_threshold=float(config.get("near_constant_threshold", 1.0e-10)),
        correlation_threshold=float(config.get("correlation_threshold", 0.95)),
    )
    if not bool(config.get("include_cycle_as_feature", False)):
        reasons[CYCLE_COLUMN] = "not configured as a multidomain feature"
    normalizer = ConditionNormalizer(
        method=str(method["normalization_method"]),
        n_regimes=int(config["operating_regime_counts"].get(method["normalization_method"], config["operating_regime_counts"]["default"])),
        random_state=int(config["random_seed"]),
        ridge_alpha=float(config["residualization"]["ridge_alpha"]),
    ).fit(fitting_frame, retained)
    frames = {"model_train": fitting_frame.copy(), **{name: frame.copy() for name, frame in apply_frames.items()}}
    for split, frame in frames.items():
        transformed = normalizer.transform(frame)
        frames[split] = transformed
    feature_columns = normalizer.output_features_
    metadata = {
        "candidate_features": candidates,
        "retained_raw_features": retained,
        "model_features": feature_columns,
        "excluded_features": reasons,
        "normalization": normalizer.metadata(),
    }
    return frames, metadata


def fit_apply_anomaly_system(
    fitting_frame: pd.DataFrame,
    apply_frames: dict[str, pd.DataFrame],
    method: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames, metadata = prepare_method_frames(fitting_frame, apply_frames, method, config)
    feature_columns = metadata["model_features"]
    model_train = frames["model_train"]
    healthy = model_train[model_train["true_rul_uncapped"] > float(config["healthy_rul_threshold"])]
    if healthy.empty:
        raise ValueError("No healthy rows are available for multidomain fitting.")
    scaler = StandardScaler().fit(healthy[feature_columns])
    transformed = {split: scaler.transform(frame[feature_columns]) for split, frame in frames.items()}
    healthy_x = scaler.transform(healthy[feature_columns])

    health_cfg = config["health_index"]
    health_model = PCAHealthIndex(
        n_components=health_cfg["n_components"],
        lower_quantile=float(health_cfg["lower_quantile"]),
        upper_quantile=float(health_cfg["upper_quantile"]),
        clip_scaled=bool(health_cfg["clip_scaled"]),
    ).fit(transformed["model_train"], model_train["true_rul_uncapped"].to_numpy(dtype=float))
    for split, frame in frames.items():
        raw, scaled = health_model.transform(transformed[split])
        frame["health_index_raw"] = raw
        frame["health_index_scaled"] = scaled
        frames[split] = smooth_by_engine(
            frame,
            value_column="health_index_scaled",
            output_column="smoothed_health_index",
            method=str(config["smoothing"]["method"]),
            window=int(config["smoothing"]["window"]),
            causal=bool(config["smoothing"]["causal"]),
        )

    det_cfg = config["detectors"]
    pca_detector = PCAReconstructionAnomalyDetector(
        n_components=det_cfg["pca_reconstruction"]["n_components"],
        threshold_percentile=float(det_cfg["pca_reconstruction"]["threshold_percentile"]),
    ).fit(healthy_x)
    iso_detector = IsolationForestAnomalyDetector(**det_cfg["isolation_forest"]).fit(healthy_x)
    svm_cfg = det_cfg["one_class_svm"]
    svm_detector = OneClassSVMAnomalyDetector(
        kernel=svm_cfg["kernel"],
        nu=float(svm_cfg["nu"]),
        gamma=svm_cfg["gamma"],
        max_training_rows=int(svm_cfg["max_healthy_training_rows"]),
        random_state=int(config["random_seed"]),
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

    calibration = config["score_calibration"]
    calibration_metadata = {"method": calibration["method"], "detectors": {}}
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
        calibration_metadata["detectors"][detector] = calibrator.metadata()

    metadata.update(
        {
            "healthy_row_count": int(len(healthy)),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "health_index_explained_variance_ratio": health_model.explained_variance_ratio_,
            "health_index_orientation": health_model.orientation_,
            "pca_reconstruction_threshold": pca_detector.threshold_,
            "isolation_forest_parameters": det_cfg["isolation_forest"],
            "one_class_svm_parameters": svm_cfg,
            "score_calibration": calibration_metadata,
        }
    )
    return frames, metadata


def evaluate_alert_frame(frame: pd.DataFrame, policy: dict[str, Any], config: dict[str, Any], label: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    result, _ = apply_alert_policy(frame, policy, config["operational_alert_thresholds"], output_prefix="locked")
    row = row_level_anomaly_metrics(result["proxy_degradation_label"], result["locked_raw_anomaly_flag"], result["locked_ensemble_score"])
    row["healthy_region_false_positive_rate"] = row["false_positive_rate"]
    critical = result[result["proxy_critical_label"] == 1]
    row["critical_region_recall"] = None if critical.empty else float(critical["locked_persistent_alarm_state"].astype(bool).mean())
    summary, engine = summarize_engine_onsets(
        result,
        detection_flag_column="locked_persistent_alarm_state",
        method_name="locked_multidomain_method",
        split_name=label,
        healthy_rul_threshold=float(config["healthy_rul_threshold"]),
        critical_rul_threshold=float(config["critical_rul_threshold"]),
    )
    engines = max(int(engine["engines_evaluated"]), 1)
    detected = int(engine["detected_engines"])
    transitions = alert_transition_metrics(result, "locked_alert_state", "locked_operational_alert_level")
    engine_context = (
        result.groupby("unit_id")
        .agg(
            global_engine_id=("global_engine_id", "first"),
            source_domain=("source_domain", "first"),
            local_unit_id=("local_unit_id", "first"),
        )
        .reset_index()
    )
    critical_recall = []
    for unit_id, group in result.groupby("unit_id"):
        critical_group = group[group["proxy_critical_label"] == 1]
        value = None if critical_group.empty else float(critical_group["locked_persistent_alarm_state"].astype(bool).mean())
        critical_recall.append({"unit_id": unit_id, "critical_region_recall": value})
    engine_context = engine_context.merge(pd.DataFrame(critical_recall), on="unit_id", how="left")
    summary = summary.merge(engine_context, on="unit_id", how="left")
    summary["alert_transition_count"] = summary["unit_id"].map(
        result.groupby("unit_id")["locked_alert_state"].apply(lambda s: sum(a != b for a, b in zip(s.astype(bool), s.astype(bool).iloc[1:])))
    )
    engine.update(
        {
            "missed_engine_rate": engine["missed_engines"] / engines,
            "false_alarm_engine_rate": engine["false_alarm_engine_count"] / engines,
            "detected_before_60_fraction": None if detected == 0 else engine["detections_before_60_cycles_rul"] / detected,
            "detected_before_30_fraction": None if detected == 0 else engine["detections_before_30_cycles_rul"] / detected,
            "mean_alert_transitions": transitions["mean_state_transitions_per_engine"],
            "median_alert_transitions": float(summary["alert_transition_count"].median()) if not summary.empty else None,
        }
    )
    return {"row_level": row, "engine_level": engine, "alert_transitions": transitions, "label": label}, summary, result


def metric_row(metrics: dict[str, Any], method: dict[str, Any], split: DomainSplit, scheme: str) -> dict[str, Any]:
    row = {
        "method_id": method["method_id"],
        "normalization_method": method["normalization_method"],
        "split_id": split.split_id,
        "validation_scheme": scheme,
        "validation_domain": ",".join(split.validation_domains),
    }
    row.update(
        {
            "precision": metrics["row_level"]["precision"],
            "recall": metrics["row_level"]["recall"],
            "f1": metrics["row_level"]["f1"],
            "healthy_region_false_positive_rate": metrics["row_level"]["healthy_region_false_positive_rate"],
            "critical_region_recall": metrics["row_level"]["critical_region_recall"],
            "engine_detection_rate": metrics["engine_level"]["detection_rate"],
            "missed_engine_rate": metrics["engine_level"]["missed_engine_rate"],
            "false_alarm_engine_rate": metrics["engine_level"]["false_alarm_engine_rate"],
            "median_lead_time": metrics["engine_level"]["median_lead_time"],
            "median_detection_delay": metrics["engine_level"]["median_detection_delay"],
            "detected_before_30_fraction": metrics["engine_level"]["detected_before_30_fraction"],
            "detected_before_60_fraction": metrics["engine_level"]["detected_before_60_fraction"],
            "mean_alert_transitions": metrics["engine_level"]["mean_alert_transitions"],
        }
    )
    return row


def evaluate_methods_on_splits(train_frame: pd.DataFrame, methods: list[dict[str, Any]], splits: list[DomainSplit], config: dict[str, Any], scheme: str) -> pd.DataFrame:
    rows = []
    for split in splits:
        fit = train_frame[train_frame["global_engine_id"].isin(split.train_engine_ids)].copy()
        validation = train_frame[train_frame["global_engine_id"].isin(split.validation_engine_ids)].copy()
        for method in methods:
            frames, _ = fit_apply_anomaly_system(fit, {"validation": validation}, method, config)
            metrics, _, _ = evaluate_alert_frame(frames["validation"], method_policy(method), config, split.split_id)
            rows.append(metric_row(metrics, method, split, scheme))
    return pd.DataFrame(rows)


def final_observed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["global_engine_id", CYCLE_COLUMN]).groupby("global_engine_id").tail(1).copy()


def fit_rul_models(train_frame: pd.DataFrame, test_frames: dict[str, pd.DataFrame], method: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, str]:
    # Reuse the locked normalization method for a simple final-cycle RUL baseline.
    candidates, retained, _, _, _ = select_anomaly_features(
        train_frame,
        include_cycle=True,
        configured_exclusions=[],
        near_constant_threshold=0.0,
        correlation_threshold=0.999,
    )
    normalizer = ConditionNormalizer(
        method=str(method["normalization_method"]),
        n_regimes=int(config["operating_regime_counts"].get(method["normalization_method"], config["operating_regime_counts"]["default"])),
        random_state=int(config["random_seed"]),
        ridge_alpha=float(config["residualization"]["ridge_alpha"]),
    ).fit(train_frame, retained)
    transformed_train = normalizer.transform(train_frame)
    features = normalizer.output_features_
    validation_ids = set(sorted(train_frame["global_engine_id"].unique())[: max(1, int(0.2 * train_frame["global_engine_id"].nunique()))])
    model_train = transformed_train[~transformed_train["global_engine_id"].isin(validation_ids)]
    validation = transformed_train[transformed_train["global_engine_id"].isin(validation_ids)]
    models = {
        "dummy_median": DummyRegressor(strategy="median"),
        "ridge": Ridge(alpha=float(config["rul_baseline"]["ridge_alpha"])),
        "random_forest": RandomForestRegressor(
            n_estimators=int(config["rul_baseline"]["random_forest"]["n_estimators"]),
            max_depth=config["rul_baseline"]["random_forest"]["max_depth"],
            min_samples_leaf=int(config["rul_baseline"]["random_forest"]["min_samples_leaf"]),
            random_state=int(config["random_seed"]),
            n_jobs=int(config["rul_baseline"]["random_forest"]["n_jobs"]),
        ),
    }
    validation_metrics = {}
    for name, model in models.items():
        model.fit(model_train[features], model_train["rul_capped"])
        pred = np.maximum(0.0, model.predict(validation[features]))
        validation_metrics[name] = regression_metrics(validation["rul_capped"], pred)
    best = min(validation_metrics, key=lambda name: validation_metrics[name]["rmse"])
    final_model = models[best]
    final_model.fit(transformed_train[features], transformed_train["rul_capped"])
    metrics: dict[str, Any] = {"validation": validation_metrics, "test": {}}
    prediction_rows = []
    for subset, raw_test in test_frames.items():
        test = normalizer.transform(raw_test)
        final_rows = final_observed_rows(test)
        pred = np.maximum(0.0, final_model.predict(final_rows[features]))
        metrics["test"][subset] = regression_metrics(final_rows["true_rul_uncapped"], pred)
        out = final_rows[["global_engine_id", "source_domain", "local_unit_id", "true_rul_uncapped"]].copy()
        out["model"] = best
        out["predicted_rul"] = pred
        out["residual"] = out["predicted_rul"] - out["true_rul_uncapped"]
        out["absolute_error"] = out["residual"].abs()
        prediction_rows.append(out)
    return metrics, pd.concat(prediction_rows, ignore_index=True), best


def bootstrap_metric(values: pd.DataFrame, metric: str, samples: int, confidence: float, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    engines = sorted(values["global_engine_id"].unique())
    if values.empty or not engines:
        return {
            "estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "valid_replicates": 0,
            "bootstrap_samples": int(samples),
            "confidence_level": float(confidence),
        }
    if metric == "rul_mae":
        estimate = float(values["absolute_error"].mean())
    elif metric == "rul_rmse":
        estimate = float(np.sqrt(np.mean(np.square(values["residual"]))))
    else:
        raise ValueError(f"Unsupported bootstrap metric: {metric}")
    estimates = []
    for _ in range(samples):
        chosen = rng.choice(engines, size=len(engines), replace=True)
        sample = pd.concat([values[values["global_engine_id"] == engine] for engine in chosen], ignore_index=True)
        if metric == "rul_mae":
            estimates.append(float(sample["absolute_error"].mean()))
        elif metric == "rul_rmse":
            estimates.append(float(np.sqrt(np.mean(np.square(sample["residual"])))))
    alpha = 1 - confidence
    return {
        "estimate": estimate,
        "ci_lower": None if not estimates else float(np.percentile(estimates, 100 * alpha / 2)),
        "ci_upper": None if not estimates else float(np.percentile(estimates, 100 * (1 - alpha / 2))),
        "valid_replicates": len(estimates),
        "bootstrap_samples": int(samples),
        "confidence_level": float(confidence),
    }


def create_figures(output_dir: Path, train: pd.DataFrame, method_ranking: pd.DataFrame, lodo: pd.DataFrame, eval_metrics: dict[str, Any], rul_metrics: dict[str, Any], domain_shift: pd.DataFrame, frames: dict[str, pd.DataFrame], config: dict[str, Any]) -> list[str]:
    figures = []
    fig_dir = output_dir / "figures"
    tl_dir = output_dir / "engine_timelines"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tl_dir.mkdir(parents=True, exist_ok=True)

    def save(path: Path) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        figures.append(str(path))

    for idx, col in enumerate(OPERATIONAL_SETTING_COLUMNS, start=1):
        plt.figure(figsize=(8, 5))
        for subset, group in train.groupby("source_domain"):
            plt.hist(group[col], bins=25, alpha=0.5, label=subset)
        plt.legend()
        plt.title(f"Operational Setting {idx} by Subset")
        save(fig_dir / f"operational_setting_{idx}_distribution_by_subset.png")
    model = OperatingRegimeModel(int(config["operating_regime_counts"]["default"]), int(config["random_seed"])).fit(train)
    sample = train.sample(min(len(train), 5000), random_state=int(config["random_seed"]))
    regimes = model.predict(sample)
    plt.figure(figsize=(7, 6))
    plt.scatter(sample[OPERATIONAL_SETTING_COLUMNS[0]], sample[OPERATIONAL_SETTING_COLUMNS[1]], c=regimes, s=5, cmap="tab10")
    plt.title("Inferred Operating Regimes")
    save(fig_dir / "inferred_operating_regimes.png")
    sensor = "sensor_2"
    plt.figure(figsize=(8, 5))
    for subset, group in train.groupby("source_domain"):
        plt.hist(group[sensor], bins=30, alpha=0.45, label=subset)
    plt.legend()
    plt.title("Sensor Distribution Before Normalization")
    save(fig_dir / "sensor_distribution_before_normalization.png")
    plt.figure(figsize=(8, 5))
    if "fd004_outside_after" in domain_shift.columns:
        plt.bar(domain_shift["subset"], domain_shift["fd004_outside_after"])
    else:
        plt.bar(domain_shift["subset"], domain_shift["outside_after"])
    plt.title("Domain Shift After Normalization")
    save(fig_dir / "domain_shift_after_normalization.png")
    plt.figure(figsize=(8, 5))
    plt.barh(method_ranking["method_id"], method_ranking["robust_utility"])
    plt.title("Validation Method Ranking")
    save(fig_dir / "validation_method_ranking.png")
    plt.figure(figsize=(8, 5))
    lodo.groupby("validation_domain")["engine_detection_rate"].mean().plot(kind="bar")
    plt.title("Leave-One-Domain-Out Performance")
    save(fig_dir / "leave_one_domain_out_performance.png")
    plt.figure(figsize=(7, 5))
    plt.scatter(method_ranking["false_alarm_engine_rate_mean"], method_ranking["engine_detection_rate_mean"])
    plt.xlabel("False-alarm rate")
    plt.ylabel("Detection rate")
    plt.title("Detection Versus False-Alarm Tradeoff")
    save(fig_dir / "detection_vs_false_alarm_tradeoff.png")
    plt.figure(figsize=(8, 5))
    labels = list(eval_metrics)
    plt.bar(labels, [eval_metrics[k]["engine_level"]["detection_rate"] for k in labels])
    plt.title("Development and External Detection Rates")
    save(fig_dir / "development_subset_comparison.png")
    plt.figure(figsize=(8, 5))
    plt.bar(labels, [eval_metrics[k]["engine_level"]["false_alarm_engine_rate"] for k in labels])
    plt.title("FD004 External and Development False Alarms")
    save(fig_dir / "fd004_external_metrics.png")
    plt.figure(figsize=(8, 5))
    plt.bar(rul_metrics["test"].keys(), [item["rmse"] for item in rul_metrics["test"].values()])
    plt.title("RUL Transfer Comparison")
    save(fig_dir / "rul_transfer_comparison.png")
    plt.figure(figsize=(8, 5))
    for subset, frame in frames.items():
        plt.hist(frame["smoothed_health_index"], bins=30, alpha=0.4, label=subset)
    plt.legend()
    plt.title("Health-Index Transfer Comparison")
    save(fig_dir / "health_index_transfer_comparison.png")
    plt.figure(figsize=(8, 5))
    for subset, frame in frames.items():
        plt.hist(frame["locked_ensemble_score"], bins=30, alpha=0.4, label=subset)
    plt.legend()
    plt.title("Detector-Score Distributions by Subset")
    save(fig_dir / "detector_score_distributions_by_subset.png")

    for subset, frame in frames.items():
        for engine in sorted(frame["global_engine_id"].unique())[: int(config["representative_timeline_count"])]:
            group = frame[frame["global_engine_id"] == engine].sort_values(CYCLE_COLUMN)
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(group[CYCLE_COLUMN], group["locked_ensemble_score"], label="score")
            axes[0].legend()
            axes[1].plot(group[CYCLE_COLUMN], group["smoothed_health_index"], color="tab:green", label="health")
            axes[1].legend()
            axes[2].plot(group[CYCLE_COLUMN], group["true_rul_uncapped"], label="true RUL")
            axes[2].step(group[CYCLE_COLUMN], group["locked_persistent_alarm_state"].astype(int) * group["true_rul_uncapped"].max(), where="post", label="alarm")
            axes[2].legend()
            path = tl_dir / f"{subset.lower()}_{engine}_timeline.png"
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close(fig)
            figures.append(str(path))
    return figures


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    start = time.perf_counter()
    root = project_root()
    config_path = Path(config_path)
    config = load_config(config_path)
    output_dir = resolve_project_path(config["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "engine_timelines").mkdir(parents=True, exist_ok=True)

    dataset_dir = resolve_project_path(config["dataset_dir"], root)
    train_raw, train_meta = load_training_subsets(
        dataset_dir,
        config["training_subsets"],
        rul_cap=float(config["healthy_rul_threshold"]),
        healthy_rul_threshold=float(config["healthy_rul_threshold"]),
        critical_rul_threshold=float(config["critical_rul_threshold"]),
    )
    train = assign_working_unit_ids(train_raw)
    dev_frames_raw, dev_meta = load_test_subsets(dataset_dir, config["development_test_subsets"], config["healthy_rul_threshold"], config["critical_rul_threshold"])
    external_raw, external_meta = load_test_subset(dataset_dir, config["external_subset"], config["healthy_rul_threshold"], config["critical_rul_threshold"])
    test_frames = {subset: assign_working_unit_ids(frame) for subset, frame in dev_frames_raw.items()}
    test_frames[str(config["external_subset"]).upper()] = assign_working_unit_ids(external_raw)

    methods = list(config["method_registry"])
    write_json(output_dir / "method_registry.json", {"methods": methods})
    splits = stratified_engine_group_splits(
        train,
        int(config["validation"]["folds"]),
        int(config["validation"]["repeats"]),
        config["validation"]["seeds"],
    )
    lodo_splits = leave_one_domain_out_splits(train, config["training_subsets"])
    validate_no_engine_leakage(splits + lodo_splits)
    validation_metrics = evaluate_methods_on_splits(train, methods, splits, config, "stratified_group_cv")
    lodo_metrics = evaluate_methods_on_splits(train, methods, lodo_splits, config, "leave_one_domain_out")
    validation_metrics.to_csv(output_dir / "validation_method_metrics.csv", index=False)
    lodo_metrics.to_csv(output_dir / "leave_one_domain_out_metrics.csv", index=False)
    combined_validation = pd.concat([validation_metrics, lodo_metrics], ignore_index=True)
    ranking = summarize_method_metrics(combined_validation, config["utility_weights"], config["feasibility_constraints"])
    ranking.to_csv(output_dir / "method_ranking.csv", index=False)
    locked_id = str(ranking.iloc[0]["method_id"])
    locked_method = next(method for method in methods if method["method_id"] == locked_id)
    write_json(
        output_dir / "locked_multidomain_method.json",
        {
            "method_id": locked_id,
            "method": locked_method,
            "selection_source": "FD001/FD002/FD003 training-engine validation only",
            "fd004_used_for_selection": False,
            "ranking_row": ranking.iloc[0].to_dict(),
        },
    )

    final_frames, final_metadata = fit_apply_anomaly_system(train, test_frames, locked_method, config)
    write_json(
        output_dir / "final_fit_metadata.json",
        {
            "training_subsets": config["training_subsets"],
            "engine_count": int(train["global_engine_id"].nunique()),
            "healthy_row_count": final_metadata["healthy_row_count"],
            "feature_set": final_metadata["model_features"],
            "normalization_method": locked_method["normalization_method"],
            "detector_settings": config["detectors"],
            "calibration_method": config["score_calibration"]["method"],
            "alert_policy": locked_method["policy"],
            "random_seed": int(config["random_seed"]),
            "fd004_used_for_fitting": False,
        },
    )
    eval_metrics: dict[str, Any] = {}
    engine_summaries = {}
    alert_frames = {}
    for subset in [*config["development_test_subsets"], config["external_subset"]]:
        subset = str(subset).upper()
        label = "untouched_external" if subset == str(config["external_subset"]).upper() else "development"
        metrics, summary, alert_frame = evaluate_alert_frame(final_frames[subset], method_policy(locked_method), config, f"{subset}_{label}")
        metrics["subset"] = subset
        metrics["evaluation_label"] = "FD004 untouched external evaluation" if label == "untouched_external" else f"{subset} development evaluation"
        eval_metrics[subset] = metrics
        engine_summaries[subset] = summary
        alert_frames[subset] = alert_frame
        if label == "untouched_external":
            write_json(output_dir / "fd004_external_metrics.json", metrics)
            summary.to_csv(output_dir / "fd004_external_engine_summary.csv", index=False)
            alert_frame.to_csv(output_dir / "fd004_cycle_level_alerts.csv", index=False)
        else:
            write_json(output_dir / f"{subset.lower()}_development_metrics.json", metrics)
            summary.to_csv(output_dir / f"{subset.lower()}_development_engine_summary.csv", index=False)

    rul_metrics, rul_predictions, locked_rul_model = fit_rul_models(train, test_frames, locked_method, config)
    write_json(output_dir / "rul_transfer_metrics.json", {"locked_model": locked_rul_model, **rul_metrics})
    rul_predictions.to_csv(output_dir / "rul_transfer_predictions.csv", index=False)

    bootstrap: dict[str, Any] = {}
    samples = int(config["bootstrap_samples"])
    confidence = float(config["confidence_level"])
    for subset, summary in engine_summaries.items():
        funcs = {
            "detection_rate": lambda frame: float(frame["detected"].astype(bool).mean()) if len(frame) else None,
            "false_alarm_engine_rate": lambda frame: float(frame["false_alarm"].astype(bool).mean()) if len(frame) else None,
            "missed_engine_rate": lambda frame: float(frame["missed"].astype(bool).mean()) if len(frame) else None,
            "median_lead_time": lambda frame: None if frame["lead_time_before_failure"].dropna().empty else float(frame["lead_time_before_failure"].dropna().median()),
            "critical_region_recall": lambda frame: None if frame["critical_region_recall"].dropna().empty else float(frame["critical_region_recall"].dropna().mean()),
        }
        bootstrap[subset] = bootstrap_engine_metrics(summary, funcs, samples, confidence, int(config["bootstrap_seed"]))
        pred_subset = rul_predictions[rul_predictions["source_domain"] == subset]
        bootstrap[subset]["rul_mae"] = bootstrap_metric(pred_subset, "rul_mae", samples, confidence, int(config["bootstrap_seed"]))
        bootstrap[subset]["rul_rmse"] = bootstrap_metric(pred_subset, "rul_rmse", samples, confidence, int(config["bootstrap_seed"]))
    write_json(output_dir / "bootstrap_confidence_intervals.json", bootstrap)

    reference = final_frames["model_train"][final_frames["model_train"]["proxy_degradation_label"] == 0]
    audit = domain_feature_audit(final_frames["model_train"], final_metadata["model_features"], reference)
    audit.to_csv(output_dir / "domain_feature_audit.csv", index=False)
    shift_rows = []
    raw_features = final_metadata["retained_raw_features"][: min(10, len(final_metadata["retained_raw_features"]))]
    model_features = final_metadata["model_features"][: min(10, len(final_metadata["model_features"]))]
    for subset, frame in alert_frames.items():
        for raw_feature, model_feature in zip(raw_features, model_features):
            before_ref = final_frames["model_train"].loc[final_frames["model_train"]["proxy_degradation_label"] == 0, raw_feature]
            before_vals = frame[raw_feature]
            after_ref = reference[model_feature]
            after_vals = frame[model_feature]
            before_std = before_ref.std(ddof=0) or 1.0
            after_std = after_ref.std(ddof=0) or 1.0
            shift_rows.append(
                {
                    "subset": subset,
                    "raw_feature": raw_feature,
                    "model_feature": model_feature,
                    "smd_before": float((before_vals.mean() - before_ref.mean()) / before_std),
                    "outside_before": float(((before_vals < before_ref.quantile(0.01)) | (before_vals > before_ref.quantile(0.99))).mean()),
                    "smd_after": float((after_vals.mean() - after_ref.mean()) / after_std),
                    "outside_after": float(((after_vals < after_ref.quantile(0.01)) | (after_vals > after_ref.quantile(0.99))).mean()),
                }
            )
    domain_shift = pd.DataFrame(shift_rows)
    domain_shift.to_csv(output_dir / "domain_shift_before_after.csv", index=False)
    shift_summary = {
        "locked_normalization_method": locked_method["normalization_method"],
        "mean_abs_smd_after_by_subset": domain_shift.assign(abs_smd=domain_shift["smd_after"].abs()).groupby("subset")["abs_smd"].mean().to_dict(),
        "reduced_shift_claim": "diagnostic only; no FD004 retuning was performed",
    }
    write_json(output_dir / "domain_shift_summary.json", shift_summary)

    fd004_metrics = eval_metrics[str(config["external_subset"]).upper()]
    fd004_rul = rul_metrics["test"][str(config["external_subset"]).upper()]
    conclusion = classify_transfer(config["generalization_criteria"], fd004_metrics, fd004_rul)
    write_json(output_dir / "generalization_conclusion.json", conclusion)

    figures = create_figures(output_dir, train, ranking, lodo_metrics, eval_metrics, rul_metrics, domain_shift, alert_frames, config)
    design_note = resolve_project_path(config.get("design_note_path", "notes/multidomain_phm_design.md"), root)
    results_note = resolve_project_path(config.get("results_note_path", "notes/multidomain_phm_results.md"), root)
    write_design_note(design_note)
    runtime = time.perf_counter() - start
    result = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "runtime_seconds": runtime,
        "training_metadata": train_meta,
        "development_test_metadata": dev_meta,
        "external_metadata": external_meta,
        "validation_folds": int(config["validation"]["folds"]),
        "validation_repeats": int(config["validation"]["repeats"]),
        "candidate_method_count": int(len(methods)),
        "locked_method": locked_method,
        "locked_rul_model": locked_rul_model,
        "validation_best_row": ranking.iloc[0].to_dict(),
        "development_metrics": {subset: eval_metrics[subset] for subset in config["development_test_subsets"]},
        "fd004_external_metrics": fd004_metrics,
        "rul_transfer_metrics": {"locked_model": locked_rul_model, **rul_metrics},
        "bootstrap_confidence_intervals": bootstrap,
        "domain_shift_findings": shift_summary,
        "generalization_conclusion": conclusion,
        "generated_files": [
            str(output_dir / "method_registry.json"),
            str(output_dir / "validation_method_metrics.csv"),
            str(output_dir / "leave_one_domain_out_metrics.csv"),
            str(output_dir / "method_ranking.csv"),
            str(output_dir / "locked_multidomain_method.json"),
            str(output_dir / "final_fit_metadata.json"),
            str(output_dir / "domain_feature_audit.csv"),
            str(output_dir / "fd001_development_metrics.json"),
            str(output_dir / "fd001_development_engine_summary.csv"),
            str(output_dir / "fd002_development_metrics.json"),
            str(output_dir / "fd002_development_engine_summary.csv"),
            str(output_dir / "fd003_development_metrics.json"),
            str(output_dir / "fd003_development_engine_summary.csv"),
            str(output_dir / "fd004_external_metrics.json"),
            str(output_dir / "fd004_external_engine_summary.csv"),
            str(output_dir / "fd004_cycle_level_alerts.csv"),
            str(output_dir / "rul_transfer_metrics.json"),
            str(output_dir / "rul_transfer_predictions.csv"),
            str(output_dir / "bootstrap_confidence_intervals.json"),
            str(output_dir / "domain_shift_before_after.csv"),
            str(output_dir / "domain_shift_summary.json"),
            str(output_dir / "generalization_conclusion.json"),
            *figures,
            str(design_note),
            str(results_note),
            str(output_dir / "run_summary.json"),
        ],
        "warnings": [
            "FD001 and FD003 test sets are previously observed development evaluations.",
            "FD002 test is a development evaluation.",
            "FD004 was evaluated only after method locking and was not used for selection or fitting.",
            "Proxy labels are RUL-threshold evaluation proxies, not certified physical anomaly labels.",
        ],
    }
    write_results_note(results_note, result, config_path)
    write_json(output_dir / "run_summary.json", result)
    print(json.dumps(_json_ready(result), indent=2, allow_nan=False))
    return result


def write_design_note(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Multidomain PHM Design

Phase 3 addresses the Phase 2C weak-generalization result by fitting a classical, operating-condition-aware PHM system across FD001, FD002, and FD003 training engines, then evaluating once on locked FD004.

FD001 and FD003 tests are previously observed development tests. FD002 test is development evaluation. FD004 test is untouched external evaluation and is not used for feature selection, normalization selection, detector fitting, calibration, method selection, RUL model selection, thresholding, persistence, or hysteresis.

The pipeline preserves local engine IDs, adds collision-safe global engine IDs, and uses engine-group validation splits. It compares bounded normalization alternatives: no adjustment, global standardization, operating-regime standardization, and continuous condition residualization. The method registry combines normalization, calibrated detector fusion, threshold, persistence, and hysteresis into fixed candidates before validation.

The locked method is selected from FD001/FD002/FD003 training-engine validation only. Final development and FD004 evaluations are direct applications of that locked system. RUL transfer remains a classical point-estimate baseline with Dummy, Ridge, and Random Forest candidates.

Bootstrap intervals resample engines, not rows. Domain-shift analysis is diagnostic and does not imply deployment certification. The implementation is original AeroGuard code and uses no deep learning, external services, package changes, or unbounded searches.
""",
        encoding="utf-8",
        newline="\n",
    )


def write_results_note(path: Path, result: dict[str, Any], config_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Multidomain PHM Results\n\n")
        handle.write(f"- Python interpreter: `{result['python_executable']}`\n")
        handle.write(f"- Python version: `{result['python_version']}`\n")
        handle.write(f"- Runtime seconds: `{result['runtime_seconds']:.3f}`\n")
        handle.write(f"- Training metadata: `{result['training_metadata']}`\n")
        handle.write(f"- Validation folds/repeats: `{result['validation_folds']}` / `{result['validation_repeats']}`\n")
        handle.write(f"- Candidate methods: `{result['candidate_method_count']}`\n")
        handle.write(f"- Locked normalization method: `{result['locked_method']['normalization_method']}`\n")
        handle.write(f"- Locked alerting policy: `{result['locked_method']['policy']}`\n")
        handle.write(f"- Locked RUL model: `{result['locked_rul_model']}`\n\n")
        handle.write("## Validation Results\n\n")
        handle.write(f"`{result['validation_best_row']}`\n\n")
        handle.write("## Development Metrics\n\n")
        handle.write(f"`{result['development_metrics']}`\n\n")
        handle.write("## FD004 External Metrics\n\n")
        handle.write(f"`{result['fd004_external_metrics']}`\n\n")
        handle.write("## RUL Transfer Metrics\n\n")
        handle.write(f"`{result['rul_transfer_metrics']}`\n\n")
        handle.write("## Bootstrap Confidence Intervals\n\n")
        handle.write(f"`{result['bootstrap_confidence_intervals']}`\n\n")
        handle.write("## Domain Shift Findings\n\n")
        handle.write(f"`{result['domain_shift_findings']}`\n\n")
        handle.write("## Generalization Conclusion\n\n")
        handle.write(f"`{result['generalization_conclusion']}`\n\n")
        handle.write("## Generated Outputs\n\n")
        for item in result["generated_files"]:
            handle.write(f"- `{item}`\n")
        handle.write("\n## Warnings and Limitations\n\n")
        for warning in result["warnings"]:
            handle.write(f"- {warning}\n")
        handle.write("\n## Exact Reproduction Command\n\n")
        handle.write("```powershell\n")
        handle.write('$env:PYTHONPATH = ".\\src"\n')
        handle.write(
            "python -m aeroguard.pipelines.train_multidomain_phm "
            f'--config "{config_path.as_posix()}"\n'
        )
        handle.write("```\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train multidomain classical PHM and evaluate locked FD004 transfer.")
    parser.add_argument("--config", required=True, help="Path to multidomain PHM YAML configuration.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
