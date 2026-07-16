"""Calibrated uncertainty-aware multidomain RUL prediction."""

from __future__ import annotations

import argparse
import hashlib
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
import sklearn
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from aeroguard.data.columns import CYCLE_COLUMN, UNIT_COLUMN
from aeroguard.data.multi_subset import load_test_subset, load_test_subsets, load_training_subsets
from aeroguard.evaluation.coverage_analysis import assign_numeric_band, bootstrap_engine_metrics, coverage_by_group
from aeroguard.evaluation.leave_one_domain_out import stratified_engine_group_splits, validate_no_engine_leakage
from aeroguard.evaluation.metrics import regression_metrics
from aeroguard.evaluation.uncertainty_metrics import interval_metrics, point_metrics
from aeroguard.features.condition_normalization import ConditionNormalizer
from aeroguard.maintenance.uncertainty_policy import assign_maintenance_recommendations, maintenance_policy_metrics
from aeroguard.pipelines.train_fd001_baseline import project_root, resolve_project_path
from aeroguard.pipelines.train_fd001_health_anomaly import select_anomaly_features
from aeroguard.pipelines.train_multidomain_phm import assign_working_unit_ids, final_observed_rows
from aeroguard.uncertainty.abstention import abstention_metrics, apply_abstention
from aeroguard.uncertainty.conformal import (
    GlobalConformalCalibrator,
    PredictedRulBandConformalCalibrator,
    assign_predicted_rul_band,
)
from aeroguard.uncertainty.interval_calibration import apply_interval_expansion, fit_interval_expansion_by_level
from aeroguard.uncertainty.quantile_regression import (
    QuantileGradientBoostingIntervals,
    quantile_gradient_boosting_available,
)
from aeroguard.uncertainty.support import SupportModel
from aeroguard.uncertainty.tree_quantiles import tree_quantile_interval_frame


REQUIRED_CONFIG_KEYS = {
    "dataset_dir",
    "training_subsets",
    "development_test_subsets",
    "external_benchmark_subset",
    "phase3_config_path",
    "phase3_results_path",
    "random_seed",
    "cross_validation_folds",
    "cross_validation_repeats",
    "cross_validation_seeds",
    "calibration_snapshot_positions",
    "minimum_snapshots_per_band",
    "nominal_coverage_levels",
    "point_model_parameters",
    "ridge_parameters",
    "random_forest_parameters",
    "quantile_gradient_boosting_parameters",
    "conformal_methods",
    "predicted_rul_bands",
    "uncertainty_method_registry",
    "maximum_method_count",
    "coverage_tolerance",
    "support_percentile_range",
    "support_threshold_candidates",
    "regime_distance_threshold_candidates",
    "interval_width_threshold_candidates",
    "abstention_rules",
    "maintenance_lower_bound_level",
    "maintenance_thresholds",
    "bootstrap_samples",
    "bootstrap_seed",
    "confidence_level",
    "calibration_classification_criteria",
    "output_dir",
    "representative_engine_count",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_config(config, project_root())
    return config


def _validate_bands(bands: list[dict[str, Any]], name: str) -> None:
    if not bands:
        raise ValueError(f"{name} must not be empty.")
    previous = -math.inf
    for band in bands:
        lower = float(band["lower"])
        upper = band.get("upper")
        upper_value = math.inf if upper is None else float(upper)
        if lower < previous or upper_value < lower:
            raise ValueError(f"Invalid {name} ordering.")
        previous = upper_value


def validate_config(config: dict[str, Any], root: Path) -> None:
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"Missing required configuration keys: {missing}")
    dataset_dir = resolve_project_path(config["dataset_dir"], root)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    training = [str(item).upper() for item in config["training_subsets"]]
    development = [str(item).upper() for item in config["development_test_subsets"]]
    external = str(config["external_benchmark_subset"]).upper()
    if external in training:
        raise ValueError("external_benchmark_subset must not be a training subset.")
    if "FD004" in training:
        raise ValueError("train_FD004.txt must not be used in Phase 4.")
    for subset in training:
        if not (dataset_dir / f"train_{subset}.txt").exists():
            raise FileNotFoundError(f"Missing training file for {subset}.")
    for subset in sorted(set(development + [external])):
        for filename in [f"test_{subset}.txt", f"RUL_{subset}.txt"]:
            if not (dataset_dir / filename).exists():
                raise FileNotFoundError(f"Missing test/RUL file: {dataset_dir / filename}")
    for path_key in ["phase3_config_path", "phase3_results_path"]:
        path = resolve_project_path(config[path_key], root)
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 3 artifact: {path}")
    folds = int(config["cross_validation_folds"])
    repeats = int(config["cross_validation_repeats"])
    if folds < 2 or repeats < 1 or len(config["cross_validation_seeds"]) != repeats:
        raise ValueError("Invalid cross-validation fold/repeat/seed configuration.")
    levels = [float(level) for level in config["nominal_coverage_levels"]]
    if any(not 0.0 < level < 1.0 for level in levels) or levels != sorted(levels):
        raise ValueError("Invalid nominal coverage levels.")
    positions = [float(item) for item in config["calibration_snapshot_positions"]]
    if any(not 0.0 < item <= 1.0 for item in positions):
        raise ValueError("Invalid calibration snapshot positions.")
    _validate_bands(config["predicted_rul_bands"], "predicted_rul_bands")
    _validate_bands(config["true_rul_bands"], "true_rul_bands")
    methods = config["uncertainty_method_registry"]
    if len(methods) > int(config["maximum_method_count"]):
        raise ValueError("Too many uncertainty methods.")
    ids = [str(method["method_id"]) for method in methods]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate uncertainty method IDs.")
    if int(config["random_forest_parameters"]["n_estimators"]) <= 0:
        raise ValueError("Invalid Random Forest parameters.")
    qgb = config["quantile_gradient_boosting_parameters"]
    if int(qgb["n_estimators"]) <= 0 or int(qgb["max_depth"]) <= 0:
        raise ValueError("Invalid Gradient Boosting parameters.")
    if float(config["coverage_tolerance"]) < 0:
        raise ValueError("Invalid coverage tolerance.")
    low, high = [float(item) for item in config["support_percentile_range"]]
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("Invalid support percentile range.")
    if int(config["bootstrap_samples"]) <= 0 or not 0.0 < float(config["confidence_level"]) < 1.0:
        raise ValueError("Invalid bootstrap configuration.")
    mt = config["maintenance_thresholds"]
    if not (float(mt["urgent_review_max"]) < float(mt["schedule_maintenance_max"]) < float(mt["plan_inspection_max"])):
        raise ValueError("Invalid maintenance-threshold ordering.")
    output_dir = resolve_project_path(config["output_dir"], root)
    for path in [output_dir, root / "notes" / "multidomain_rul_uncertainty_design.md", root / "notes" / "multidomain_rul_uncertainty_results.md"]:
        lowered = str(path).lower()
        if "\\references\\" in lowered or "\\extracted-code\\" in lowered:
            raise ValueError("Outputs must not be inside protected folders.")


def environment_report() -> dict[str, Any]:
    ok, reason = quantile_gradient_boosting_available()
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "scikit_learn_version": sklearn.__version__,
        "gradient_boosting_regressor_available": True,
        "gradient_boosting_quantile_loss_available": ok,
        "gradient_boosting_quantile_loss_reason": reason,
        "random_forest_available": True,
        "standard_scaler_available": True,
    }


def create_phase3_manifest(output_dir: Path, root: Path, config: dict[str, Any]) -> dict[str, Any]:
    phase3_dir = resolve_project_path(config["phase3_results_path"], root)
    important = [
        phase3_dir / "locked_multidomain_method.json",
        phase3_dir / "final_fit_metadata.json",
        phase3_dir / "fd004_external_metrics.json",
        phase3_dir / "rul_transfer_metrics.json",
        phase3_dir / "generalization_conclusion.json",
        phase3_dir / "method_ranking.csv",
        phase3_dir / "validation_method_metrics.csv",
        phase3_dir / "leave_one_domain_out_metrics.csv",
    ]
    hashes = {str(path): sha256_file(path) for path in important if path.exists()}
    fd004 = json.loads((phase3_dir / "fd004_external_metrics.json").read_text(encoding="utf-8"))
    rul = json.loads((phase3_dir / "rul_transfer_metrics.json").read_text(encoding="utf-8"))
    conclusion = json.loads((phase3_dir / "generalization_conclusion.json").read_text(encoding="utf-8"))
    locked = json.loads((phase3_dir / "locked_multidomain_method.json").read_text(encoding="utf-8"))
    manifest = {
        "phase3_artifact_paths": [str(path) for path in important],
        "locked_alert_method": locked["method_id"],
        "locked_normalization_method": locked["method"]["normalization_method"],
        "locked_point_rul_model": rul["locked_model"],
        "fd004_phase3_rul_mae": rul["test"]["FD004"]["mae"],
        "fd004_phase3_rul_rmse": rul["test"]["FD004"]["rmse"],
        "fd004_phase3_anomaly_metrics": fd004,
        "phase3_generalization_conclusion": conclusion,
        "sha256": hashes,
        "statement": "Phase 3 results were read for manifesting only and were not overwritten.",
    }
    write_json(output_dir / "phase3_benchmark_manifest.json", manifest)
    return manifest


def select_snapshots(frame: pd.DataFrame, positions: list[float]) -> pd.DataFrame:
    rows = []
    for _, group in frame.sort_values(["global_engine_id", CYCLE_COLUMN]).groupby("global_engine_id"):
        max_cycle = float(group[CYCLE_COLUMN].max())
        for position in positions:
            target = position * max_cycle
            idx = (group[CYCLE_COLUMN].astype(float) - target).abs().idxmin()
            row = group.loc[idx].copy()
            row["normalized_life_position"] = float(position)
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def fit_bundle(train: pd.DataFrame, config: dict[str, Any], fit_quantile: bool) -> dict[str, Any]:
    _, retained, reasons, _, _ = select_anomaly_features(
        train,
        include_cycle=True,
        configured_exclusions=list(config["point_model_parameters"].get("features_to_exclude", [])),
        near_constant_threshold=float(config["point_model_parameters"]["near_constant_threshold"]),
        correlation_threshold=float(config["point_model_parameters"]["correlation_threshold"]),
    )
    normalizer = ConditionNormalizer(
        method=str(config["point_model_parameters"]["normalization_method"]),
        n_regimes=int(config["point_model_parameters"]["operating_regime_count"]),
        random_state=int(config["random_seed"]),
        ridge_alpha=float(config["point_model_parameters"]["residualization_ridge_alpha"]),
    ).fit(train, retained)
    transformed = normalizer.transform(train)
    features = normalizer.output_features_
    scaler = StandardScaler().fit(transformed[features])
    x_train = scaler.transform(transformed[features])
    y_train = transformed["rul_capped"].to_numpy(dtype=float)
    rf_params = dict(config["random_forest_parameters"])
    rf_params["random_state"] = int(config["random_seed"])
    rf = RandomForestRegressor(**rf_params).fit(x_train, y_train)
    ridge = Ridge(**dict(config["ridge_parameters"])).fit(x_train, y_train)
    qgb_model = None
    qgb_ok, qgb_reason = quantile_gradient_boosting_available()
    if fit_quantile and qgb_ok and bool(config["quantile_gradient_boosting_parameters"].get("enabled", True)):
        qgb_params = dict(config["quantile_gradient_boosting_parameters"])
        qgb_params.pop("enabled", None)
        qgb_model = QuantileGradientBoostingIntervals(
            nominal_levels=[float(level) for level in config["nominal_coverage_levels"]],
            parameters=qgb_params,
            random_state=int(config["random_seed"]),
        ).fit(x_train, y_train)
    low, high = [float(item) for item in config["support_percentile_range"]]
    support = SupportModel(
        feature_columns=features,
        percentile_low=low,
        percentile_high=high,
        feature_exceedance_limited=float(config["support_threshold_candidates"]["limited_feature_exceedance"]),
        feature_exceedance_out=float(config["support_threshold_candidates"]["out_feature_exceedance"]),
        robust_distance_limited=float(config["support_threshold_candidates"]["limited_robust_distance"]),
        robust_distance_out=float(config["support_threshold_candidates"]["out_robust_distance"]),
        regime_distance_quantile=float(config["regime_distance_threshold_candidates"]["quantile"]),
    ).fit(transformed)
    return {
        "normalizer": normalizer,
        "scaler": scaler,
        "features": features,
        "retained_raw_features": retained,
        "excluded_features": reasons,
        "rf": rf,
        "ridge": ridge,
        "qgb": qgb_model,
        "qgb_reason": qgb_reason,
        "support": support,
    }


def predict_bundle(bundle: dict[str, Any], frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    transformed = bundle["normalizer"].transform(frame)
    features = bundle["features"]
    x_values = bundle["scaler"].transform(transformed[features])
    levels = [float(level) for level in config["nominal_coverage_levels"]]
    output = transformed[[
        "subset",
        "source_domain",
        "global_engine_id",
        "local_unit_id",
        UNIT_COLUMN,
        CYCLE_COLUMN,
        "true_rul_uncapped",
        "proxy_health_region",
    ]].copy()
    output["operating_regime"] = transformed["operating_regime"] if "operating_regime" in transformed.columns else -1
    if "normalized_life_position" in transformed.columns:
        output["normalized_life_position"] = transformed["normalized_life_position"].astype(float)
    output["true_rul"] = output["true_rul_uncapped"].astype(float)
    output["predicted_rul"] = np.maximum(0.0, bundle["rf"].predict(x_values))
    output["ridge_predicted_rul"] = np.maximum(0.0, bundle["ridge"].predict(x_values))
    tree = tree_quantile_interval_frame(bundle["rf"], x_values, levels, prefix="tree_")
    output = pd.concat([output.reset_index(drop=True), tree.reset_index(drop=True)], axis=1)
    if bundle.get("qgb") is not None:
        qgb = bundle["qgb"].predict_interval_frame(x_values, prefix="qgb_")
        output = pd.concat([output.reset_index(drop=True), qgb.reset_index(drop=True)], axis=1)
    else:
        for level in levels:
            pct = int(round(level * 100))
            output[f"qgb_lower_{pct}"] = output[f"tree_lower_{pct}"]
            output[f"qgb_upper_{pct}"] = output[f"tree_upper_{pct}"]
            output[f"qgb_quantile_crossing_{pct}"] = False
        output["qgb_quantile_crossing_any"] = False
    support = bundle["support"].score(transformed)
    output = pd.concat([output.reset_index(drop=True), support.reset_index(drop=True)], axis=1)
    output["residual"] = output["predicted_rul"] - output["true_rul"]
    output["absolute_error"] = output["residual"].abs()
    output["ridge_residual"] = output["ridge_predicted_rul"] - output["true_rul"]
    output["ridge_absolute_error"] = output["ridge_residual"].abs()
    return output


def add_intervals_for_method(
    frame: pd.DataFrame,
    method: dict[str, Any],
    calibrators: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    result = frame.copy()
    levels = [float(level) for level in config["nominal_coverage_levels"]]
    kind = str(method["interval_method"])
    point = str(method["point_model"])
    point_col = "ridge_predicted_rul" if point == "ridge" else "predicted_rul"
    if kind in {"global_grouped_conformal", "cross_conformal"}:
        interval_frame = calibrators["global"].interval_frame(result[point_col])
        result = pd.concat([result.reset_index(drop=True), interval_frame.reset_index(drop=True)], axis=1)
    elif kind == "predicted_rul_band_conformal":
        interval_frame = calibrators["band"].interval_frame(result[point_col])
        result = pd.concat([result.reset_index(drop=True), interval_frame.reset_index(drop=True)], axis=1)
    elif kind == "rf_tree_quantile":
        for level in levels:
            pct = int(round(level * 100))
            result[f"lower_{pct}"] = result[f"tree_lower_{pct}"]
            result[f"upper_{pct}"] = result[f"tree_upper_{pct}"]
            result[f"raw_lower_{pct}"] = result[f"tree_raw_lower_{pct}"]
    elif kind == "calibrated_rf_tree_quantile":
        tmp = result.rename(columns={f"tree_lower_{int(round(level * 100))}": f"lower_{int(round(level * 100))}" for level in levels})
        tmp = tmp.rename(columns={f"tree_upper_{int(round(level * 100))}": f"upper_{int(round(level * 100))}" for level in levels})
        result = apply_interval_expansion(tmp, calibrators["tree_corrections"])
    elif kind == "quantile_gradient_boosting":
        for level in levels:
            pct = int(round(level * 100))
            result[f"lower_{pct}"] = result[f"qgb_lower_{pct}"]
            result[f"upper_{pct}"] = result[f"qgb_upper_{pct}"]
            result[f"raw_lower_{pct}"] = result.get(f"qgb_raw_lower_{pct}", result[f"qgb_lower_{pct}"])
    elif kind == "conformalized_quantile_gradient_boosting":
        tmp = result.rename(columns={f"qgb_lower_{int(round(level * 100))}": f"lower_{int(round(level * 100))}" for level in levels})
        tmp = tmp.rename(columns={f"qgb_upper_{int(round(level * 100))}": f"upper_{int(round(level * 100))}" for level in levels})
        result = apply_interval_expansion(tmp, calibrators["qgb_corrections"])
    elif kind == "ridge_global_conformal":
        interval_frame = calibrators["ridge_global"].interval_frame(result["ridge_predicted_rul"])
        result = pd.concat([result.reset_index(drop=True), interval_frame.reset_index(drop=True)], axis=1)
    else:
        raise ValueError(f"Unsupported interval method: {kind}")
    result["method_id"] = method["method_id"]
    result["point_model"] = point
    if point_col != "predicted_rul":
        result["predicted_rul"] = result[point_col]
        result["residual"] = result["predicted_rul"] - result["true_rul"]
        result["absolute_error"] = result["residual"].abs()
    for level in levels:
        pct = int(round(level * 100))
        result[f"interval_width_{pct}"] = result[f"upper_{pct}"].astype(float) - result[f"lower_{pct}"].astype(float)
        result[f"covered_{pct}"] = (result["true_rul"] >= result[f"lower_{pct}"]) & (result["true_rul"] <= result[f"upper_{pct}"])
    return result


def fit_calibrators(oof: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    levels = [float(level) for level in config["nominal_coverage_levels"]]
    calibrators: dict[str, Any] = {
        "global": GlobalConformalCalibrator(levels).fit(oof["residual"]),
        "band": PredictedRulBandConformalCalibrator(
            levels,
            config["predicted_rul_bands"],
            int(config["minimum_snapshots_per_band"]),
        ).fit(oof["predicted_rul"], oof["residual"]),
        "ridge_global": GlobalConformalCalibrator(levels).fit(oof["ridge_residual"]),
    }
    tree_base = oof.rename(columns={f"tree_lower_{int(round(level * 100))}": f"tree_lower_{int(round(level * 100))}" for level in levels})
    calibrators["tree_corrections"] = fit_interval_expansion_by_level(
        tree_base.rename(
            columns={
                **{f"tree_lower_{int(round(level * 100))}": f"tree_lower_{int(round(level * 100))}" for level in levels},
                **{f"tree_upper_{int(round(level * 100))}": f"tree_upper_{int(round(level * 100))}" for level in levels},
            }
        ).assign(true_rul=oof["true_rul"]),
        levels,
        prefix="tree_",
    )
    calibrators["qgb_corrections"] = fit_interval_expansion_by_level(
        oof.assign(true_rul=oof["true_rul"]),
        levels,
        prefix="qgb_",
    )
    return calibrators


def method_metrics(frame: pd.DataFrame, method: dict[str, Any], config: dict[str, Any], split_label: str) -> list[dict[str, Any]]:
    rows = []
    point = point_metrics(frame["true_rul"], frame["predicted_rul"])
    for level in config["nominal_coverage_levels"]:
        pct = int(round(float(level) * 100))
        metrics = interval_metrics(frame["true_rul"], frame["predicted_rul"], frame[f"lower_{pct}"], frame[f"upper_{pct}"], float(level))
        rows.append(
            {
                "method_id": method["method_id"],
                "point_model": method["point_model"],
                "interval_method": method["interval_method"],
                "split_label": split_label,
                **point,
                **metrics,
            }
        )
    return rows


def rank_methods(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    level = float(config["selection_nominal_level"])
    tolerance = float(config["coverage_tolerance"])
    rows = []
    selected = metrics[metrics["nominal_level"] == level]
    for method_id, group in selected.groupby("method_id"):
        row = group.iloc[0].to_dict()
        row["feasible"] = bool(row["coverage"] >= level - tolerance)
        row["selection_score"] = (
            (0.0 if row["feasible"] else 10.0)
            + max(0.0, level - float(row["coverage"])) * 20.0
            + float(row["mean_interval_width"]) / 100.0
            + float(row["mae"]) / 100.0
            + float(row["absolute_coverage_error"])
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["feasible", "selection_score", "mean_interval_width"], ascending=[False, True, True])


def make_cv_outputs(train: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    start = time.perf_counter()
    splits = stratified_engine_group_splits(
        train,
        int(config["cross_validation_folds"]),
        int(config["cross_validation_repeats"]),
        config["cross_validation_seeds"],
    )
    validate_no_engine_leakage(splits)
    rows = []
    split_records = []
    positions = [float(item) for item in config["calibration_snapshot_positions"]]
    qgb_enabled = any("quantile" in str(method["interval_method"]) for method in config["uncertainty_method_registry"])
    for split in splits:
        fit = train[train["global_engine_id"].isin(split.train_engine_ids)].copy()
        held = train[train["global_engine_id"].isin(split.validation_engine_ids)].copy()
        bundle = fit_bundle(fit, config, fit_quantile=qgb_enabled)
        snapshots = select_snapshots(held, positions)
        predictions = predict_bundle(bundle, snapshots, config)
        predictions["fold"] = split.split_id
        predictions["repeat"] = split.split_id.split("_")[1] if "_" in split.split_id else split.split_id
        predictions["predicted_rul_band"] = assign_predicted_rul_band(predictions["predicted_rul"], config["predicted_rul_bands"])
        rows.append(predictions)
        split_records.append(
            {
                **split.to_dict(),
                "train_engine_count": len(split.train_engine_ids),
                "validation_engine_count": len(split.validation_engine_ids),
                "train_subset_counts": fit.groupby("source_domain")["global_engine_id"].nunique().to_dict(),
                "validation_subset_counts": held.groupby("source_domain")["global_engine_id"].nunique().to_dict(),
                "engine_overlap": sorted(set(split.train_engine_ids).intersection(split.validation_engine_ids)),
            }
        )
    return pd.concat(rows, ignore_index=True), split_records, {"no_engine_overlap": True, "split_count": len(splits)}, {"cross_validation_seconds": time.perf_counter() - start}


def final_prediction_frame(train: pd.DataFrame, test_frames: dict[str, pd.DataFrame], config: dict[str, Any], calibrators: dict[str, Any], locked_method: dict[str, Any], oof_width90: float) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    start = time.perf_counter()
    qgb_needed = "quantile" in str(locked_method["interval_method"]) or any("quantile" in str(method["interval_method"]) for method in config["uncertainty_method_registry"])
    bundle = fit_bundle(train, config, fit_quantile=qgb_needed)
    rows = []
    for subset, frame in test_frames.items():
        final_rows = final_observed_rows(frame)
        pred = predict_bundle(bundle, final_rows, config)
        pred["subset"] = subset
        with_intervals = add_intervals_for_method(pred, locked_method, calibrators, config)
        with_intervals["final_observed_cycle"] = with_intervals[CYCLE_COLUMN]
        rows.append(with_intervals)
    predictions = pd.concat(rows, ignore_index=True)
    predictions["predicted_rul_band"] = assign_predicted_rul_band(predictions["predicted_rul"], config["predicted_rul_bands"])
    predictions["true_rul_band"] = assign_numeric_band(predictions["true_rul"], config["true_rul_bands"], "true_rul_band")
    predictions["trajectory_length_group"] = pd.cut(
        predictions["final_observed_cycle"],
        bins=[0, 100, 200, 10000],
        labels=["short", "medium", "long"],
        include_lowest=True,
    ).astype(str)
    predictions["point_error_group"] = pd.cut(
        predictions["absolute_error"],
        bins=[-0.001, 10, 30, math.inf],
        labels=["low_error", "medium_error", "high_error"],
    ).astype(str)
    predictions["interval_width_ratio"] = predictions["interval_width_90"] / max(float(oof_width90), 1.0e-9)
    predictions = apply_abstention(predictions, config["abstention_rules"])
    predictions = assign_maintenance_recommendations(predictions, config["maintenance_thresholds"], "lower_90")
    metadata = {
        "feature_set": bundle["features"],
        "retained_raw_features": bundle["retained_raw_features"],
        "excluded_features": bundle["excluded_features"],
        "normalization": bundle["normalizer"].metadata(),
        "support": bundle["support"].metadata(),
        "random_forest_parameters": config["random_forest_parameters"],
        "ridge_parameters": config["ridge_parameters"],
        "quantile_gradient_boosting": None if bundle["qgb"] is None else bundle["qgb"].metadata(),
        "fd004_used_for_fitting_calibration_or_selection": False,
    }
    return predictions, metadata, {"final_fit_seconds": time.perf_counter() - start}


def calibration_and_coverage_outputs(predictions: pd.DataFrame, metrics: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    levels = [float(level) for level in config["nominal_coverage_levels"]]
    calibration: dict[str, Any] = {
        "cross_validation": metrics.to_dict(orient="records"),
        "test_by_subset": {},
    }
    for subset, group in predictions.groupby("subset"):
        rows = method_metrics(group, {"method_id": "locked", "point_model": "random_forest", "interval_method": "locked"}, config, subset)
        calibration["test_by_subset"][subset] = rows
    coverage_tables = {
        "subset": coverage_by_group(predictions, "subset", levels),
        "regime": coverage_by_group(predictions, "operating_regime", levels),
        "rul_band": coverage_by_group(predictions, "true_rul_band", levels),
        "support_status": coverage_by_group(predictions, "support_status", levels),
    }
    return calibration, coverage_tables


def build_bootstrap(predictions: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    output = {}
    levels = [int(round(float(level) * 100)) for level in config["nominal_coverage_levels"]]
    for subset, group in predictions.groupby("subset"):
        def mae(frame: pd.DataFrame) -> float:
            return float(frame["absolute_error"].mean())

        def rmse(frame: pd.DataFrame) -> float:
            return float(np.sqrt(np.mean(np.square(frame["residual"]))))

        funcs = {
            "mae": mae,
            "rmse": rmse,
            "mean_90_interval_width": lambda f: float(f["interval_width_90"].mean()),
            "median_90_interval_width": lambda f: float(f["interval_width_90"].median()),
            "abstention_rate": lambda f: float(f["abstain_flag"].astype(bool).mean()),
            "accepted_prediction_mae": lambda f: None if f[~f["abstain_flag"]].empty else float(f.loc[~f["abstain_flag"], "absolute_error"].mean()),
            "urgent_review_recall_true_rul_le_15": lambda f: None
            if (f["true_rul"] <= 15).sum() == 0
            else float(((f["true_rul"] <= 15) & f["maintenance_action"].isin(["URGENT_ENGINEERING_REVIEW", "ENGINEERING_REVIEW_REQUIRED"])).sum() / (f["true_rul"] <= 15).sum()),
        }
        for pct in levels:
            funcs[f"coverage_{pct}"] = lambda f, pct=pct: float(f[f"covered_{pct}"].astype(bool).mean())
        output[subset] = bootstrap_engine_metrics(
            group,
            funcs,
            int(config["bootstrap_samples"]),
            float(config["confidence_level"]),
            int(config["bootstrap_seed"]),
        )
    return output


def classify_calibration(metrics: pd.DataFrame, predictions: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    criteria = config["calibration_classification_criteria"]
    cv90 = float(metrics[(metrics["nominal_level"] == 0.90) & (metrics["method_id"] == config["_locked_method_id"])]["coverage"].iloc[0])
    fd004 = predictions[predictions["subset"] == str(config["external_benchmark_subset"]).upper()]
    fd004_cov90 = float(fd004["covered_90"].mean())
    fd004_width90 = float(fd004["interval_width_90"].mean())
    abstention = float(predictions["abstain_flag"].mean())
    values = {
        "cv_coverage_90": cv90,
        "fd004_coverage_90": fd004_cov90,
        "fd004_mean_width_90": fd004_width90,
        "abstention_rate": abstention,
    }
    for label in ["well_calibrated", "moderately_calibrated", "weakly_calibrated"]:
        rule = criteria[label]
        failed = []
        if cv90 < float(rule["min_cv_coverage_90"]):
            failed.append("cv_coverage_90")
        if fd004_cov90 < float(rule["min_fd004_coverage_90"]):
            failed.append("fd004_coverage_90")
        if fd004_width90 > float(rule["max_fd004_mean_width_90"]):
            failed.append("fd004_mean_width_90")
        if abstention > float(rule["max_abstention_rate"]):
            failed.append("abstention_rate")
        if not failed:
            return {"classification": label.replace("_", " ").title(), "criteria_values": values, "failed_criteria": failed}
    return {"classification": "Failed Calibration Transfer", "criteria_values": values, "failed_criteria": ["No configured criteria were satisfied."]}


def create_figures(output_dir: Path, predictions: pd.DataFrame, metrics: pd.DataFrame, coverage_tables: dict[str, pd.DataFrame], oof: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    fig_dir = output_dir / "figures"
    ex_dir = output_dir / "engine_examples"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ex_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []

    def save(path: Path) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        figures.append(str(path))

    plt.figure(figsize=(8, 6))
    plt.errorbar(predictions["true_rul"], predictions["predicted_rul"], yerr=[predictions["predicted_rul"] - predictions["lower_90"], predictions["upper_90"] - predictions["predicted_rul"]], fmt=".", alpha=0.35)
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    save(fig_dir / "predicted_vs_true_rul_90_intervals.png")

    plt.figure(figsize=(7, 5))
    levels = [int(round(float(level) * 100)) for level in config["nominal_coverage_levels"]]
    plt.plot(levels, [predictions[f"covered_{pct}"].mean() for pct in levels], marker="o", label="empirical")
    plt.plot(levels, [pct / 100 for pct in levels], linestyle="--", label="nominal")
    plt.legend()
    save(fig_dir / "coverage_vs_nominal_confidence.png")

    for column, filename, xlabel in [
        ("true_rul", "interval_width_vs_true_rul.png", "True RUL"),
        ("predicted_rul", "interval_width_vs_predicted_rul.png", "Predicted RUL"),
    ]:
        plt.figure(figsize=(7, 5))
        plt.scatter(predictions[column], predictions["interval_width_90"], s=12, alpha=0.45)
        plt.xlabel(xlabel)
        plt.ylabel("90% interval width")
        save(fig_dir / filename)

    for key, filename in [
        ("subset", "coverage_by_subset.png"),
        ("rul_band", "coverage_by_rul_band.png"),
        ("regime", "coverage_by_operating_regime.png"),
        ("support_status", "coverage_by_support_status.png"),
    ]:
        table = coverage_tables[key]
        group_col = table.columns[0]
        plt.figure(figsize=(8, 5))
        plt.bar(table[group_col].astype(str), table["coverage_90"])
        plt.xticks(rotation=30, ha="right")
        save(fig_dir / filename)

    plt.figure(figsize=(8, 5))
    selected = metrics[metrics["nominal_level"] == 0.90]
    plt.scatter(selected["mean_interval_width"], selected["coverage"])
    for _, row in selected.iterrows():
        plt.text(row["mean_interval_width"], row["coverage"], row["method_id"], fontsize=7)
    plt.xlabel("Mean width")
    plt.ylabel("CV coverage")
    save(fig_dir / "method_coverage_width_tradeoff.png")

    plt.figure(figsize=(8, 5))
    selected["coverage"].plot(kind="hist", bins=12)
    save(fig_dir / "cross_validation_coverage_distribution.png")

    plt.figure(figsize=(8, 5))
    predictions["residual"].plot(kind="hist", bins=30)
    save(fig_dir / "residual_distribution.png")

    plt.figure(figsize=(7, 5))
    predictions.groupby("support_status")["abstain_flag"].mean().plot(kind="bar")
    save(fig_dir / "abstention_tradeoff.png")

    plt.figure(figsize=(7, 5))
    before = predictions["absolute_error"].mean()
    accepted = predictions.loc[~predictions["abstain_flag"], "absolute_error"].mean()
    plt.bar(["before", "accepted"], [before, accepted])
    save(fig_dir / "error_before_after_abstention.png")

    plt.figure(figsize=(9, 5))
    predictions["maintenance_action"].value_counts().plot(kind="bar")
    plt.xticks(rotation=30, ha="right")
    save(fig_dir / "maintenance_action_distribution.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(predictions["true_rul"], predictions["lower_90"], s=12, alpha=0.5)
    plt.xlabel("True RUL")
    plt.ylabel("Conservative lower 90% bound")
    save(fig_dir / "conservative_lower_bound_vs_true_rul.png")

    examples = predictions.sort_values("absolute_error").head(int(config["representative_engine_count"]))
    extra = pd.concat(
        [
            examples,
            predictions.sort_values("interval_width_90", ascending=False).head(1),
            predictions[~predictions["covered_90"]].head(1),
            predictions[predictions["abstain_flag"]].head(1),
            predictions[predictions["subset"] == str(config["external_benchmark_subset"]).upper()].head(1),
            predictions[predictions["true_rul"] <= 15].head(1),
        ],
        ignore_index=True,
    ).drop_duplicates("global_engine_id")
    for _, row in extra.iterrows():
        plt.figure(figsize=(6, 4))
        plt.errorbar([0], [row["predicted_rul"]], yerr=[[row["predicted_rul"] - row["lower_90"]], [row["upper_90"] - row["predicted_rul"]]], fmt="o")
        plt.axhline(row["true_rul"], color="tab:green", linestyle="--", label="true RUL")
        plt.title(str(row["global_engine_id"]))
        plt.legend()
        save(ex_dir / f"{row['global_engine_id']}_uncertainty_example.png")
    return figures


def write_design_note(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Multidomain RUL Uncertainty Design

Phase 4 extends the Phase 3 multidomain RUL system from point estimates to calibrated prediction intervals, support scoring, abstention, and conservative demonstration maintenance recommendations.

FD001, FD002, and FD003 training engines are used for point-model fitting, engine-group cross-validation, uncertainty-method selection, conformal calibration, support-threshold selection, abstention configuration, and maintenance-policy configuration. FD001-FD003 tests are development evaluations. FD004 is a previously evaluated external benchmark and is used only after the Phase 4 method is locked.

The pipeline uses engine-group splits, engine-balanced normalized-life calibration snapshots, grouped conformal residual intervals, predicted-RUL-band conformal intervals, Random Forest tree quantiles, Quantile Gradient Boosting, conformalized intervals, support analysis, and abstention. Coverage is evaluated against sharpness, and undercoverage is treated as the safety-relevant failure mode.

Maintenance recommendations are demonstration decision-support outputs based on the lower 90% RUL bound and abstention status. They are not approved aircraft-maintenance instructions.

Known limitations include C-MAPSS simulation limits, capped-target training, tree-quantile undercoverage risk under domain shift, and the fact that conformal guarantees depend on calibration exchangeability that may not hold across FD004.
""",
        encoding="utf-8",
        newline="\n",
    )


def write_results_note(path: Path, result: dict[str, Any], config_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Multidomain RUL Uncertainty Results\n\n")
        for key in [
            "python_executable",
            "python_version",
            "scikit_learn_version",
            "training_engine_counts",
            "test_engine_counts",
            "cross_validation",
            "calibration_snapshot_count",
            "candidate_method_count",
            "locked_point_model",
            "locked_uncertainty_method",
            "selection_rationale",
            "calibration_conclusion",
            "runtime_seconds",
            "runtime_by_stage",
        ]:
            handle.write(f"- {key}: `{result.get(key)}`\n")
        handle.write("\n## Metrics\n\n")
        handle.write(f"- Cross-validation interval metrics: `{result['cross_validation_metrics_summary']}`\n")
        handle.write(f"- Test metrics by subset: `{result['test_metrics_by_subset']}`\n")
        handle.write(f"- Abstention metrics: `{result['abstention_metrics']}`\n")
        handle.write(f"- Maintenance metrics: `{result['maintenance_policy_metrics']}`\n")
        handle.write(f"- Bootstrap confidence intervals: `{result['bootstrap_confidence_intervals']}`\n")
        handle.write("\n## Generated Files\n\n")
        for item in result["generated_files"]:
            handle.write(f"- `{item}`\n")
        handle.write("\n## Warnings\n\n")
        for warning in result["warnings"]:
            handle.write(f"- {warning}\n")
        handle.write("\n## Reproduction Command\n\n```powershell\n")
        handle.write('$env:PYTHONPATH = ".\\src"\n')
        handle.write(
            "python -m aeroguard.pipelines.train_multidomain_rul_uncertainty "
            f'--config "{config_path.as_posix()}"\n'
        )
        handle.write("```\n")


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    start = time.perf_counter()
    stage_times: dict[str, float] = {}
    root = project_root()
    config_path = Path(config_path)
    config = load_config(config_path)
    output_dir = resolve_project_path(config["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "engine_examples").mkdir(parents=True, exist_ok=True)
    env = environment_report()
    manifest = create_phase3_manifest(output_dir, root, config)

    load_start = time.perf_counter()
    dataset_dir = resolve_project_path(config["dataset_dir"], root)
    train_raw, train_meta = load_training_subsets(
        dataset_dir,
        config["training_subsets"],
        rul_cap=float(config["point_model_parameters"]["rul_cap"]),
        healthy_rul_threshold=float(config["point_model_parameters"]["healthy_rul_threshold"]),
        critical_rul_threshold=float(config["point_model_parameters"]["critical_rul_threshold"]),
    )
    train = assign_working_unit_ids(train_raw)
    dev_frames_raw, dev_meta = load_test_subsets(
        dataset_dir,
        config["development_test_subsets"],
        config["point_model_parameters"]["healthy_rul_threshold"],
        config["point_model_parameters"]["critical_rul_threshold"],
    )
    external_raw, external_meta = load_test_subset(
        dataset_dir,
        config["external_benchmark_subset"],
        config["point_model_parameters"]["healthy_rul_threshold"],
        config["point_model_parameters"]["critical_rul_threshold"],
    )
    test_frames = {subset: assign_working_unit_ids(frame) for subset, frame in dev_frames_raw.items()}
    test_frames[str(config["external_benchmark_subset"]).upper()] = assign_working_unit_ids(external_raw)
    stage_times["data_loading_seconds"] = time.perf_counter() - load_start

    registry = list(config["uncertainty_method_registry"])
    qgb_ok, qgb_reason = quantile_gradient_boosting_available()
    if not qgb_ok:
        registry = [method for method in registry if "quantile" not in str(method["interval_method"])]
    write_json(output_dir / "uncertainty_method_registry.json", {"methods": registry, "qgb_availability": qgb_reason})

    oof, split_records, leakage_report, cv_times = make_cv_outputs(train, {**config, "uncertainty_method_registry": registry})
    stage_times.update(cv_times)
    snapshots_out = oof[
        [
            "fold",
            "repeat",
            "subset",
            "global_engine_id",
            "local_unit_id",
            CYCLE_COLUMN,
            "normalized_life_position",
            "true_rul",
            "predicted_rul",
            "residual",
            "absolute_error",
            "predicted_rul_band",
            "operating_regime",
        ]
    ].rename(columns={CYCLE_COLUMN: "cycle"})
    snapshots_out.to_csv(output_dir / "calibration_snapshots.csv", index=False)
    write_json(output_dir / "cross_validation_splits.json", {"splits": split_records, "leakage_report": leakage_report})

    cal_start = time.perf_counter()
    calibrators = fit_calibrators(oof, config)
    metric_rows = []
    method_frames: dict[str, pd.DataFrame] = {}
    for method in registry:
        frame = add_intervals_for_method(oof, method, calibrators, config)
        method_frames[method["method_id"]] = frame
        metric_rows.extend(method_metrics(frame, method, config, "cross_validation_snapshots"))
    cv_metrics = pd.DataFrame(metric_rows)
    ranking = rank_methods(cv_metrics, config)
    locked_id = str(ranking.iloc[0]["method_id"])
    locked_method = next(method for method in registry if method["method_id"] == locked_id)
    config["_locked_method_id"] = locked_id
    cv_metrics.to_csv(output_dir / "cross_validation_uncertainty_metrics.csv", index=False)
    ranking.to_csv(output_dir / "uncertainty_method_ranking.csv", index=False)
    write_json(
        output_dir / "locked_uncertainty_method.json",
        {
            "method": locked_method,
            "selection_source": "FD001/FD002/FD003 training-engine cross-validation snapshots only",
            "fd004_used_for_selection": False,
            "ranking_row": ranking.iloc[0].to_dict(),
        },
    )
    locked_oof = method_frames[locked_id]
    oof_width90 = float(locked_oof["interval_width_90"].median())
    stage_times["calibration_selection_seconds"] = time.perf_counter() - cal_start

    predictions, fit_metadata, fit_times = final_prediction_frame(train, test_frames, config, calibrators, locked_method, oof_width90)
    stage_times.update(fit_times)
    output_columns = [
        "subset",
        UNIT_COLUMN,
        "global_engine_id",
        "final_observed_cycle",
        "operating_regime",
        "true_rul",
        "predicted_rul",
        "residual",
        "absolute_error",
        "lower_80",
        "upper_80",
        "covered_80",
        "lower_90",
        "upper_90",
        "covered_90",
        "lower_95",
        "upper_95",
        "covered_95",
        "interval_width_80",
        "interval_width_90",
        "interval_width_95",
        "support_status",
        "support_score",
        "abstain_flag",
        "abstention_reason",
        "maintenance_action",
        "prediction_status",
        "feature_exceedance_fraction",
        "regime_distance",
        "interval_width_ratio",
    ]
    predictions[output_columns].to_csv(output_dir / "uncertainty_predictions.csv", index=False)
    write_json(
        output_dir / "final_uncertainty_fit_metadata.json",
        {
            **fit_metadata,
            "locked_uncertainty_method": locked_method,
            "conformal_calibrators": {
                "global": calibrators["global"].metadata(),
                "band": calibrators["band"].metadata(),
                "ridge_global": calibrators["ridge_global"].metadata(),
                "tree_corrections": calibrators["tree_corrections"],
                "qgb_corrections": calibrators["qgb_corrections"],
            },
        },
    )
    write_json(output_dir / "support_thresholds.json", fit_metadata["support"])

    calibration, coverage_tables = calibration_and_coverage_outputs(predictions, cv_metrics, config)
    write_json(output_dir / "calibration_metrics.json", calibration)
    coverage_tables["subset"].to_csv(output_dir / "coverage_by_subset.csv", index=False)
    coverage_tables["regime"].to_csv(output_dir / "coverage_by_regime.csv", index=False)
    coverage_tables["rul_band"].to_csv(output_dir / "coverage_by_rul_band.csv", index=False)
    coverage_tables["support_status"].to_csv(output_dir / "coverage_by_support_status.csv", index=False)

    abstention = {
        subset: abstention_metrics(group, level=90, high_error_threshold=float(config["abstention_rules"]["high_error_threshold"]))
        for subset, group in predictions.groupby("subset")
    }
    abstention_analysis = pd.DataFrame(
        [
            {"subset": subset, **metrics}
            for subset, metrics in abstention.items()
        ]
    )
    abstention_analysis.to_csv(output_dir / "abstention_analysis.csv", index=False)
    write_json(output_dir / "abstention_metrics.json", abstention)

    maintenance_recs = predictions[[
        "subset",
        "global_engine_id",
        UNIT_COLUMN,
        "true_rul",
        "predicted_rul",
        "lower_90",
        "upper_90",
        "abstain_flag",
        "maintenance_action",
        "action_basis",
        "conservative_rul_bound",
        "nominal_interval_level",
        "prediction_status",
        "maintenance_disclaimer",
    ]]
    maintenance_recs.to_csv(output_dir / "maintenance_recommendations.csv", index=False)
    maintenance_metrics = maintenance_policy_metrics(predictions)
    write_json(output_dir / "maintenance_policy_metrics.json", maintenance_metrics)
    bootstrap = build_bootstrap(predictions, config)
    write_json(output_dir / "bootstrap_confidence_intervals.json", bootstrap)
    conclusion = classify_calibration(cv_metrics, predictions, config)
    write_json(output_dir / "calibration_conclusion.json", conclusion)

    figures = create_figures(output_dir, predictions, cv_metrics, coverage_tables, oof, config)
    design_note = root / "notes" / "multidomain_rul_uncertainty_design.md"
    results_note = root / "notes" / "multidomain_rul_uncertainty_results.md"
    write_design_note(design_note)

    runtime = time.perf_counter() - start
    stage_times["total_runtime_seconds"] = runtime
    test_metrics = {
        subset: {
            "point": point_metrics(group["true_rul"], group["predicted_rul"]),
            "intervals": {
                str(level): interval_metrics(group["true_rul"], group["predicted_rul"], group[f"lower_{int(round(level * 100))}"], group[f"upper_{int(round(level * 100))}"], level)
                for level in config["nominal_coverage_levels"]
            },
        }
        for subset, group in predictions.groupby("subset")
    }
    generated_files = [
        str(output_dir / name)
        for name in [
            "phase3_benchmark_manifest.json",
            "cross_validation_splits.json",
            "calibration_snapshots.csv",
            "uncertainty_method_registry.json",
            "cross_validation_uncertainty_metrics.csv",
            "uncertainty_method_ranking.csv",
            "locked_uncertainty_method.json",
            "final_uncertainty_fit_metadata.json",
            "support_thresholds.json",
            "uncertainty_predictions.csv",
            "calibration_metrics.json",
            "coverage_by_subset.csv",
            "coverage_by_regime.csv",
            "coverage_by_rul_band.csv",
            "coverage_by_support_status.csv",
            "abstention_analysis.csv",
            "abstention_metrics.json",
            "maintenance_recommendations.csv",
            "maintenance_policy_metrics.json",
            "bootstrap_confidence_intervals.json",
            "calibration_conclusion.json",
            "run_summary.json",
        ]
    ]
    generated_files.extend(figures)
    generated_files.extend([str(design_note), str(results_note)])
    result = {
        **env,
        "runtime_seconds": runtime,
        "runtime_by_stage": stage_times,
        "training_metadata": train_meta,
        "development_test_metadata": dev_meta,
        "external_benchmark_metadata": external_meta,
        "training_engine_counts": train.groupby("source_domain")["global_engine_id"].nunique().to_dict(),
        "test_engine_counts": {subset: int(frame["global_engine_id"].nunique()) for subset, frame in test_frames.items()},
        "cross_validation": {
            "folds": int(config["cross_validation_folds"]),
            "repeats": int(config["cross_validation_repeats"]),
            "seeds": config["cross_validation_seeds"],
            "leakage_report": leakage_report,
        },
        "calibration_snapshot_count": int(len(oof)),
        "candidate_method_count": int(len(registry)),
        "locked_point_model": locked_method["point_model"],
        "locked_uncertainty_method": locked_method,
        "selection_rationale": ranking.iloc[0].to_dict(),
        "cross_validation_metrics_summary": cv_metrics.to_dict(orient="records"),
        "test_metrics_by_subset": test_metrics,
        "abstention_metrics": abstention,
        "maintenance_policy_metrics": maintenance_metrics,
        "bootstrap_confidence_intervals": bootstrap,
        "calibration_conclusion": conclusion,
        "phase3_manifest": manifest,
        "generated_files": generated_files,
        "warnings": [
            "FD004 is a previously evaluated external benchmark, not untouched.",
            "FD004 was not used for fitting, calibration, threshold selection, abstention selection, maintenance-policy configuration, or method selection.",
            "Intervals and recommendations are research demonstration outputs, not certified aircraft-maintenance instructions.",
            "Tree quantiles are empirical dispersion summaries and are not automatically calibrated uncertainty.",
        ],
    }
    write_results_note(results_note, result, config_path)
    write_json(output_dir / "run_summary.json", result)
    print(json.dumps(_json_ready(result), indent=2, allow_nan=False))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train multidomain RUL uncertainty system.")
    parser.add_argument("--config", required=True, help="Path to Phase 4 RUL uncertainty YAML config.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
