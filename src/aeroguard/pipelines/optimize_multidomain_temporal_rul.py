"""Phase 5B temporal RUL model optimization."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
import sklearn

from aeroguard.data.columns import CYCLE_COLUMN, UNIT_COLUMN
from aeroguard.data.multi_subset import load_test_subsets, load_training_subsets
from aeroguard.deep.checkpoints import save_checkpoint
from aeroguard.deep.extended_training import train_for_fixed_epochs, train_with_early_stopping
from aeroguard.deep.models import MODEL_CLASSES
from aeroguard.deep.models.common import trainable_parameter_count, validate_parameter_budget
from aeroguard.deep.reproducibility import set_global_seed
from aeroguard.deep.sampling import build_endpoint_table
from aeroguard.deep.seed_evaluation import aggregate_seed_metrics, prediction_disagreement
from aeroguard.deep.sequence_dataset import SequenceWindowDataset
from aeroguard.deep.windowing import WindowSpec, final_endpoint_table, sequence_audit
from aeroguard.evaluation.coverage_analysis import assign_numeric_band
from aeroguard.evaluation.deep_rul_metrics import metrics_by_group, prediction_direction
from aeroguard.evaluation.leave_one_domain_out import stratified_engine_group_splits, validate_no_engine_leakage
from aeroguard.evaluation.model_efficiency import model_efficiency_row
from aeroguard.evaluation.temporal_model_stability import locked_epoch_from_cv, summarize_model_stability
from aeroguard.evaluation.uncertainty_metrics import interval_metrics
from aeroguard.maintenance.uncertainty_policy import assign_maintenance_recommendations, maintenance_policy_metrics
from aeroguard.pipelines.train_fd001_baseline import project_root, resolve_project_path
from aeroguard.pipelines.train_multidomain_deep_rul import (
    _json_ready,
    add_uncertainty,
    apply_preprocessor,
    apply_support_abstention_maintenance,
    environment_report,
    evaluate_model_frame,
    fit_preprocessor,
    make_dataset,
    point_metrics_for_predictions,
    screening_split,
    sha256_file,
    snapshot_endpoint_table,
    write_json,
)
from aeroguard.pipelines.train_multidomain_phm import assign_working_unit_ids
from aeroguard.uncertainty.abstention import abstention_metrics
from aeroguard.uncertainty.conformal import GlobalConformalCalibrator, PredictedRulBandConformalCalibrator


REQUIRED_CONFIG_KEYS = {
    "dataset_dir",
    "training_subsets",
    "benchmark_test_subsets",
    "phase5_config_path",
    "phase5_results_path",
    "phase5_checkpoint_path",
    "random_seed",
    "execution_profile",
    "device",
    "deterministic_algorithms",
    "window_length",
    "window_stride",
    "minimum_valid_history",
    "maximum_windows_per_engine",
    "sampling_method",
    "validation_snapshot_positions",
    "rul_bands",
    "training_target",
    "rul_cap",
    "healthy_rul_threshold",
    "critical_rul_threshold",
    "include_cycle_as_feature",
    "features_to_exclude",
    "near_constant_threshold",
    "correlation_threshold",
    "operating_condition_method",
    "number_of_operating_regimes",
    "residualization_ridge_alpha",
    "screening_validation_fraction",
    "screening_seed",
    "maximum_candidate_count",
    "finalist_count",
    "finalist_cv_folds",
    "finalist_cv_seed",
    "finalist_seeds",
    "maximum_seed_run_count",
    "training_schedules",
    "optimizer",
    "weight_decay",
    "loss",
    "gradient_clip_norm",
    "batch_size",
    "num_workers",
    "mixed_precision",
    "parameter_budget",
    "latency_feasibility_threshold_ms",
    "robust_selection_weights",
    "improvement_classification",
    "transformer_defaults",
    "patch_options",
    "model_registry",
    "nominal_coverage_levels",
    "conformal_methods",
    "predicted_rul_bands",
    "coverage_tolerance",
    "support_settings",
    "abstention_settings",
    "maintenance_thresholds",
    "checkpoint_dir",
    "output_dir",
    "representative_engine_count",
    "plotting",
}


def _validate_bands(bands: list[dict[str, Any]]) -> None:
    previous = -math.inf
    for band in bands:
        lower = float(band["lower"])
        upper = band.get("upper")
        upper_value = math.inf if upper is None else float(upper)
        if lower < previous or upper_value < lower:
            raise ValueError("Invalid RUL band ordering.")
        previous = upper_value


def _validate_thresholds(thresholds: dict[str, Any]) -> None:
    if not float(thresholds["urgent_review_max"]) < float(thresholds["schedule_maintenance_max"]) < float(thresholds["plan_inspection_max"]):
        raise ValueError("Invalid maintenance thresholds.")


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
    valid = {"FD001", "FD002", "FD003", "FD004"}
    for subset in [str(item).upper() for item in config["training_subsets"]]:
        if subset not in valid:
            raise ValueError(f"Invalid training subset: {subset}")
        if not (dataset_dir / f"train_{subset}.txt").exists():
            raise FileNotFoundError(f"Missing training file for {subset}.")
    for subset in [str(item).upper() for item in config["benchmark_test_subsets"]]:
        if subset not in valid:
            raise ValueError(f"Invalid benchmark subset: {subset}")
        for name in [f"test_{subset}.txt", f"RUL_{subset}.txt"]:
            if not (dataset_dir / name).exists():
                raise FileNotFoundError(f"Missing benchmark file: {name}")
    for key in ["phase5_config_path", "phase5_results_path", "phase5_checkpoint_path"]:
        if not resolve_project_path(config[key], root).exists():
            raise FileNotFoundError(f"Missing Phase 5 artifact: {config[key]}")
    if not (resolve_project_path(config["phase5_results_path"], root) / "run_summary.json").exists():
        raise FileNotFoundError("Missing Phase 5 run_summary.json.")
    if int(config["window_length"]) <= 0 or int(config["window_stride"]) <= 0:
        raise ValueError("Invalid sequence settings.")
    if not 1 <= int(config["minimum_valid_history"]) <= int(config["window_length"]):
        raise ValueError("Invalid minimum_valid_history.")
    if int(config["maximum_windows_per_engine"]) <= 0:
        raise ValueError("Invalid maximum_windows_per_engine.")
    if int(config["maximum_candidate_count"]) < 1 or len(config["model_registry"]) > int(config["maximum_candidate_count"]):
        raise ValueError("Invalid candidate count.")
    if int(config["finalist_count"]) < 1 or int(config["finalist_cv_folds"]) < 2:
        raise ValueError("Invalid finalist count or fold count.")
    if not config["finalist_seeds"] or int(config["maximum_seed_run_count"]) < 1:
        raise ValueError("Invalid seed count.")
    _validate_bands(config["rul_bands"])
    _validate_bands(config["predicted_rul_bands"])
    for level in config["nominal_coverage_levels"]:
        if not 0.0 < float(level) < 1.0:
            raise ValueError("Invalid nominal coverage.")
    if set(config["conformal_methods"]) - {"global_grouped_conformal", "predicted_rul_band_conformal"}:
        raise ValueError("Invalid conformal method.")
    _validate_thresholds(config["maintenance_thresholds"])
    for schedule_id, schedule in config["training_schedules"].items():
        if int(schedule["max_epochs"]) <= 0 or int(schedule["minimum_epochs"]) <= 0 or int(schedule["early_stopping_patience"]) < 0:
            raise ValueError(f"Invalid epoch or patience values in {schedule_id}.")
        if int(schedule["minimum_epochs"]) > int(schedule["max_epochs"]):
            raise ValueError(f"minimum_epochs exceeds max_epochs in {schedule_id}.")
        if float(schedule["learning_rate"]) <= 0:
            raise ValueError(f"Invalid learning rate in {schedule_id}.")
        if schedule["scheduler"] not in {"plateau", "cosine", "none"}:
            raise ValueError(f"Invalid scheduler in {schedule_id}.")
    model_ids = [str(item["model_id"]) for item in config["model_registry"]]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Duplicate model IDs.")
    for candidate in config["model_registry"]:
        architecture = str(candidate["architecture"])
        if architecture not in MODEL_CLASSES or architecture not in {"lstm", "tcn", "temporal_transformer", "patch_transformer"}:
            raise ValueError(f"Unsupported Phase 5B architecture: {architecture}")
        if candidate["schedule_id"] not in config["training_schedules"]:
            raise ValueError(f"Invalid schedule for {candidate['model_id']}")
        if architecture in {"temporal_transformer", "patch_transformer"}:
            projection = int(candidate["projection_dim"])
            heads = int(candidate["heads"])
            if projection % heads != 0:
                raise ValueError("Projection dimension not divisible by attention heads.")
            if int(candidate["layers"]) <= 0:
                raise ValueError("Invalid Transformer layer count.")
            if candidate["positional_encoding"] not in {"sinusoidal", "learnable"}:
                raise ValueError("Invalid positional encoding.")
            if candidate["pooling"] not in {"mean", "attention", "final"}:
                raise ValueError("Invalid pooling method.")
        if architecture == "patch_transformer":
            if int(candidate["patch_length"]) <= 0 or int(candidate["patch_stride"]) <= 0:
                raise ValueError("Invalid patch length or stride.")
            if int(candidate["patch_length"]) > int(config["window_length"]):
                raise ValueError("Patch length greater than window length.")
    output_dir = resolve_project_path(config["output_dir"], root)
    checkpoint_dir = resolve_project_path(config["checkpoint_dir"], root)
    for path in [output_dir, checkpoint_dir, root / "notes" / "multidomain_temporal_optimization_design.md", root / "notes" / "multidomain_temporal_optimization_results.md"]:
        lowered = str(path).lower()
        if "\\references\\" in lowered or "\\extracted-code\\" in lowered:
            raise ValueError("Outputs must not be inside protected directories.")


def select_execution_profile(config: dict[str, Any], env: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    requested = str(config["execution_profile"])
    if requested == "auto":
        if env["cuda_available"]:
            total = int((env.get("gpu_memory") or {}).get("total_bytes", 0) or 0)
            profile = "gpu_laptop_safe" if total and total <= 8 * 1024**3 else "gpu_laptop_safe"
            reason = "CUDA is available; bounded laptop-safe Phase 5B limits are active."
        else:
            profile = "cpu_safe"
            reason = "CUDA is unavailable."
    else:
        profile = requested
        reason = "Profile explicitly configured."
    limits = dict(config["execution_limits"].get(profile, {}))
    if not limits:
        raise ValueError(f"Unsupported execution profile: {profile}")
    limits["reason"] = limits.get("reason", reason)
    limits["profile"] = profile
    limits["effective_finalist_count"] = min(int(config["finalist_count"]), int(limits["finalist_count"]))
    limits["effective_finalist_seed_count"] = min(len(config["finalist_seeds"]), int(limits["finalist_seed_count"]))
    limits["effective_mixed_precision"] = bool(env["cuda_available"] and config["mixed_precision"] != "false" and profile != "cpu_safe")
    return profile, limits


def device_from_config(config: dict[str, Any], env: dict[str, Any]) -> torch.device:
    requested = str(config["device"])
    if requested == "auto":
        return torch.device("cuda" if env["cuda_available"] else "cpu")
    return torch.device(requested)


def schedule_for_candidate(config: dict[str, Any], candidate: dict[str, Any], limits: dict[str, Any], stage: str) -> dict[str, Any]:
    schedule = dict(config["training_schedules"][candidate["schedule_id"]])
    epoch_cap = int(limits["stage_a_epoch_cap"] if stage == "stage_a" else limits["finalist_epoch_cap"])
    schedule["configured_max_epochs"] = int(schedule["max_epochs"])
    schedule["max_epochs"] = min(int(schedule["max_epochs"]), epoch_cap)
    schedule["minimum_epochs"] = min(int(schedule["minimum_epochs"]), int(schedule["max_epochs"]))
    schedule.update(
        {
            "optimizer": config["optimizer"],
            "weight_decay": float(config["weight_decay"]),
            "loss": config["loss"],
            "gradient_clip_norm": float(config["gradient_clip_norm"]),
            "batch_size": int(config["batch_size"]),
            "num_workers": int(config["num_workers"]),
            "pin_memory": bool(config.get("pin_memory", False)),
            "severe_optimistic_threshold": float(config["severe_optimistic_threshold"]),
        }
    )
    return schedule


def build_model(candidate: dict[str, Any], input_dim: int, config: dict[str, Any]) -> torch.nn.Module:
    architecture = str(candidate["architecture"])
    params = {key: value for key, value in candidate.items() if key not in {"model_id", "architecture", "schedule_id"}}
    if architecture in {"temporal_transformer", "patch_transformer"}:
        params.setdefault("max_length", int(config["window_length"]))
    if architecture == "patch_transformer":
        params.setdefault("window_length", int(config["window_length"]))
    model = MODEL_CLASSES[architecture](input_dim=input_dim, **params)
    validate_parameter_budget(model, int(config["parameter_budget"]["default"]))
    return model


def train_candidate(
    candidate: dict[str, Any],
    train_dataset: SequenceWindowDataset,
    validation_dataset: SequenceWindowDataset,
    validation_metadata: pd.DataFrame,
    config: dict[str, Any],
    schedule: dict[str, Any],
    device: torch.device,
    input_dim: int,
    seed: int,
    mixed_precision: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    set_global_seed(seed, bool(config.get("deterministic_algorithms", False)))
    model = build_model(candidate, input_dim, config)
    try:
        return train_with_early_stopping(model, train_dataset, validation_dataset, schedule, device, validation_metadata, mixed_precision)
    except RuntimeError as exc:
        if not mixed_precision or "Non-finite" not in str(exc):
            raise
        set_global_seed(seed, bool(config.get("deterministic_algorithms", False)))
        model = build_model(candidate, input_dim, config)
        model, metadata = train_with_early_stopping(model, train_dataset, validation_dataset, schedule, device, validation_metadata, False)
        metadata["mixed_precision_retry_reason"] = "disabled_after_non_finite_training"
        return model, metadata


def train_locked(
    candidate: dict[str, Any],
    train_dataset: SequenceWindowDataset,
    config: dict[str, Any],
    schedule: dict[str, Any],
    device: torch.device,
    input_dim: int,
    seed: int,
    epochs: int,
    mixed_precision: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    set_global_seed(seed, bool(config.get("deterministic_algorithms", False)))
    model = build_model(candidate, input_dim, config)
    try:
        return train_for_fixed_epochs(model, train_dataset, schedule, device, epochs, mixed_precision)
    except RuntimeError as exc:
        if not mixed_precision or "Non-finite" not in str(exc):
            raise
        set_global_seed(seed, bool(config.get("deterministic_algorithms", False)))
        model = build_model(candidate, input_dim, config)
        model, metadata = train_for_fixed_epochs(model, train_dataset, schedule, device, epochs, False)
        metadata["mixed_precision_retry_reason"] = "disabled_after_non_finite_training"
        return model, metadata


def checkpoint_size_mb(path: Path) -> float:
    return float(path.stat().st_size / (1024.0 * 1024.0))


def create_phase5_manifest(output_dir: Path, root: Path, config: dict[str, Any]) -> dict[str, Any]:
    phase5 = resolve_project_path(config["phase5_results_path"], root)
    checkpoint = resolve_project_path(config["phase5_checkpoint_path"], root)
    summary = json.loads((phase5 / "run_summary.json").read_text(encoding="utf-8"))
    files = [
        phase5 / "run_summary.json",
        phase5 / "benchmark_predictions.csv",
        phase5 / "benchmark_metrics.json",
        phase5 / "classical_vs_deep.csv",
        phase5 / "deep_uncertainty_predictions.csv",
        phase5 / "deep_uncertainty_metrics.json",
        phase5 / "model_efficiency.csv",
        phase5 / "locked_deep_model.json",
        checkpoint,
    ]
    hashes = {str(path): sha256_file(path) for path in files if path.exists()}
    efficiency = pd.read_csv(phase5 / "model_efficiency.csv") if (phase5 / "model_efficiency.csv").exists() else pd.DataFrame()
    locked = summary["locked_architecture"]
    eff_row = efficiency[efficiency["model_id"] == locked].iloc[0].to_dict() if not efficiency.empty and locked in set(efficiency["model_id"]) else {}
    manifest = {
        "phase5_results_path": str(phase5),
        "phase5_checkpoint_path": str(checkpoint),
        "important_report_paths": [str(path) for path in files if path.parent == phase5],
        "important_checkpoint_paths": [str(checkpoint)],
        "locked_phase5_architecture": locked,
        "locked_phase5_epoch_count": summary["locked_epoch_count"],
        "metrics_by_subset": summary["benchmark_metrics"],
        "fd004_mae": summary["benchmark_metrics"]["FD004"]["mae"],
        "fd004_rmse": summary["benchmark_metrics"]["FD004"]["rmse"],
        "fd004_90_coverage": summary["deep_uncertainty_metrics"]["FD004"]["0.9"]["coverage"],
        "deep_uncertainty_method": summary["locked_deep_uncertainty_method"]["method_id"],
        "parameter_count": eff_row.get("parameter_count"),
        "model_size_bytes": eff_row.get("serialized_size_bytes"),
        "cpu_latency_ms": eff_row.get("cpu_batch_one_median_latency_ms"),
        "gpu_latency_ms": eff_row.get("gpu_batch_one_median_latency_ms"),
        "sha256": hashes,
        "statement": "Phase 5 artifacts were read only and were not modified by Phase 5B.",
    }
    write_json(output_dir / "phase5_benchmark_manifest.json", manifest)
    return manifest


def choose_uncertainty_method(oof: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    levels = [float(level) for level in config["nominal_coverage_levels"]]
    global_cal = GlobalConformalCalibrator(levels).fit(oof["residual"])
    band_cal = PredictedRulBandConformalCalibrator(levels, config["predicted_rul_bands"], minimum_samples_per_band=30).fit(oof["predicted_rul"], oof["residual"])
    candidates = {
        "phase5b_global_grouped_conformal": global_cal,
        "phase5b_predicted_band_conformal": band_cal,
    }
    rows = []
    for method_id, calibrator in candidates.items():
        frame = add_uncertainty(oof, calibrator, method_id, levels)
        for level in levels:
            pct = int(round(level * 100))
            metrics = interval_metrics(frame["true_rul"], frame["predicted_rul"], frame[f"lower_{pct}"], frame[f"upper_{pct}"], level)
            variability = frame.groupby(["fold", "seed"])[f"covered_{pct}"].mean().std(ddof=0) if {"fold", "seed"}.issubset(frame.columns) else 0.0
            rows.append({"uncertainty_method_id": method_id, "coverage_std": 0.0 if pd.isna(variability) else float(variability), **metrics})
    metrics_df = pd.DataFrame(rows)
    selected = metrics_df[metrics_df["nominal_level"] == 0.90].copy()
    selected["feasible"] = selected["coverage"] >= 0.90 - float(config["coverage_tolerance"])
    selected = selected.sort_values(
        ["feasible", "mean_interval_width", "mean_interval_score", "coverage_std", "uncertainty_method_id"],
        ascending=[False, True, True, True, True],
    )
    locked = str(selected.iloc[0]["uncertainty_method_id"])
    return {"method_id": locked, "calibrator": candidates[locked].metadata(), "selection_source": "training-engine CV only"}, metrics_df, {"locked_method_id": locked, "global": global_cal, "band": band_cal}


def _metric_rows_by_subset(frame: pd.DataFrame, model_name: str, metrics: dict[str, Any], extra: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for subset, subset_metrics in metrics.items():
        if subset == "overall":
            continue
        rows.append({"subset": subset, "model": model_name, **subset_metrics, **extra})
    return rows


def compare_phase5_phase5b(output_dir: Path, root: Path, config: dict[str, Any], benchmark_metrics: dict[str, Any], locked: dict[str, Any], final_meta: dict[str, Any], efficiency: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    phase5_summary = json.loads((resolve_project_path(config["phase5_results_path"], root) / "run_summary.json").read_text(encoding="utf-8"))
    rows = []
    rows.extend(_metric_rows_by_subset(phase5_summary["benchmark_metrics"], "phase5_lstm", phase5_summary["benchmark_metrics"], {"architecture": "lstm", "training_schedule": "phase5", "locked_epoch_count": phase5_summary["locked_epoch_count"]}))
    eff_row = efficiency[efficiency["model_id"] == locked["model_id"]].iloc[0].to_dict() if not efficiency.empty and locked["model_id"] in set(efficiency["model_id"]) else {}
    rows.extend(_metric_rows_by_subset(benchmark_metrics, "phase5b_locked", benchmark_metrics, {"architecture": locked["architecture"], "training_schedule": locked["schedule_id"], "locked_epoch_count": locked["locked_epoch_count"], "parameter_count": locked["parameter_count"], "checkpoint_size_mb": locked["checkpoint_size_mb"], "training_runtime_seconds": final_meta["training_seconds"], "cpu_latency_ms": eff_row.get("cpu_batch_one_median_latency_ms"), "gpu_latency_ms": eff_row.get("gpu_batch_one_median_latency_ms")}))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "phase5_vs_phase5b.csv", index=False)
    base = phase5_summary["benchmark_metrics"]["overall"]["rmse"]
    new = benchmark_metrics["overall"]["rmse"]
    fd004_base = phase5_summary["benchmark_metrics"]["FD004"]["rmse"]
    fd004_new = benchmark_metrics["FD004"]["rmse"]
    rmse_gain = max((base - new) / max(base, 1.0e-9), (fd004_base - fd004_new) / max(fd004_base, 1.0e-9))
    crit = config["improvement_classification"]
    if rmse_gain >= float(crit["clear_min_rmse_reduction_fraction"]):
        conclusion = "Clear Phase 5B improvement"
    elif rmse_gain >= float(crit["moderate_min_rmse_reduction_fraction"]):
        conclusion = "Moderate Phase 5B improvement"
    elif abs(rmse_gain) <= float(crit["comparable_abs_rmse_delta_fraction"]):
        conclusion = "Comparable performance"
    elif rmse_gain < 0:
        conclusion = "Phase 5 LSTM remains stronger"
    else:
        conclusion = "Inconclusive"
    return comparison, conclusion


def uncertainty_comparison(output_dir: Path, root: Path, config: dict[str, Any], phase5b_uncertainty: pd.DataFrame, uncertainty_metrics: dict[str, Any]) -> pd.DataFrame:
    phase5_summary = json.loads((resolve_project_path(config["phase5_results_path"], root) / "run_summary.json").read_text(encoding="utf-8"))
    phase5_predictions = pd.read_csv(resolve_project_path(config["phase5_results_path"], root) / "deep_uncertainty_predictions.csv")
    rows = []
    for subset in [item for item in uncertainty_metrics if item != "overall"]:
        for model_name, source, frame in [
            ("phase5", phase5_summary["deep_uncertainty_metrics"][subset], phase5_predictions[phase5_predictions["subset"] == subset]),
            ("phase5b", uncertainty_metrics[subset], phase5b_uncertainty[phase5b_uncertainty["subset"] == subset]),
        ]:
            accepted = frame[~frame["abstain_flag"].astype(bool)] if "abstain_flag" in frame.columns else frame
            rows.append(
                {
                    "subset": subset,
                    "model": model_name,
                    "coverage_80": source["0.8"]["coverage"],
                    "coverage_90": source["0.9"]["coverage"],
                    "coverage_95": source["0.95"]["coverage"],
                    "mean_width_90": source["0.9"]["mean_interval_width"],
                    "median_width_90": source["0.9"]["median_interval_width"],
                    "undercoverage_90": source["0.9"]["undercoverage_amount"],
                    "interval_score_90": source["0.9"]["mean_interval_score"],
                    "abstention_rate": float(frame["abstain_flag"].mean()) if "abstain_flag" in frame.columns else 0.0,
                    "accepted_prediction_mae": None if accepted.empty else float(accepted["absolute_error"].mean()),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "phase5_vs_phase5b_uncertainty.csv", index=False)
    return result


def make_figures(
    output_dir: Path,
    screening: pd.DataFrame,
    cv: pd.DataFrame,
    stability: pd.DataFrame,
    benchmark: pd.DataFrame,
    comparison: pd.DataFrame,
    uncertainty: pd.DataFrame,
    uncertainty_comparison_frame: pd.DataFrame,
    efficiency: pd.DataFrame,
) -> list[str]:
    figures: list[str] = []
    fig_dir = output_dir / "figures"
    ex_dir = output_dir / "engine_examples"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ex_dir.mkdir(parents=True, exist_ok=True)

    def save(path: Path) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        figures.append(str(path))

    for model_id, group in screening.groupby("model_id"):
        if "history" in group.columns:
            history = group.iloc[0]["history"]
            if isinstance(history, list):
                hist = pd.DataFrame(history)
                plt.figure(figsize=(8, 5)); plt.plot(hist["epoch"], hist["train_loss"], label="train"); plt.plot(hist["epoch"], hist["validation_loss"], label="validation"); plt.legend(); save(fig_dir / f"training_curve_{model_id}.png")
    plt.figure(figsize=(9, 5)); screening.set_index("model_id")["best_epoch"].plot(kind="bar"); plt.xticks(rotation=30, ha="right"); save(fig_dir / "best_epoch_by_candidate.png")
    plt.figure(figsize=(9, 5)); screening.set_index("model_id")["validation_rmse"].plot(kind="bar"); plt.xticks(rotation=30, ha="right"); save(fig_dir / "stage_a_validation_rmse.png")
    plt.figure(figsize=(9, 5)); screening.set_index("model_id")["validation_nasa_score"].plot(kind="bar"); plt.xticks(rotation=30, ha="right"); save(fig_dir / "stage_a_nasa_score.png")
    plt.figure(figsize=(8, 5)); cv.boxplot(column="validation_rmse", by="model_id", ax=plt.gca()); plt.suptitle(""); plt.xticks(rotation=30, ha="right"); save(fig_dir / "finalist_cv_rmse_distributions.png")
    plt.figure(figsize=(8, 5)); cv.groupby(["model_id", "seed"])["validation_rmse"].mean().unstack(0).plot(kind="bar", ax=plt.gca()); save(fig_dir / "seed_stability_comparison.png")
    plt.figure(figsize=(7, 5)); plt.scatter(stability["mean_rmse"], stability["std_rmse"]); plt.xlabel("Mean RMSE"); plt.ylabel("RMSE std"); save(fig_dir / "mean_rmse_vs_variability.png")
    plt.figure(figsize=(7, 5)); plt.scatter(efficiency["parameter_count"], efficiency["validation_rmse"]); plt.xlabel("Parameters"); plt.ylabel("Validation RMSE"); save(fig_dir / "parameter_count_vs_rmse.png")
    plt.figure(figsize=(7, 5)); plt.scatter(efficiency["cpu_batch_one_median_latency_ms"], efficiency["validation_rmse"]); plt.xlabel("CPU latency ms"); plt.ylabel("Validation RMSE"); save(fig_dir / "cpu_latency_vs_rmse.png")
    plt.figure(figsize=(8, 5)); comparison.pivot(index="subset", columns="model", values="rmse").plot(kind="bar", ax=plt.gca()); save(fig_dir / "phase5_vs_phase5b_metrics_by_subset.png")
    plt.figure(figsize=(7, 5)); plt.scatter(benchmark["true_rul"], benchmark["predicted_rul"], s=12, alpha=0.5); plt.xlabel("True RUL"); plt.ylabel("Predicted RUL"); save(fig_dir / "predicted_vs_true_rul.png")
    plt.figure(figsize=(8, 5)); benchmark["residual"].plot(kind="hist", bins=30); save(fig_dir / "residual_distribution.png")
    for column, name in [("true_rul_band", "error_by_rul_band.png"), ("operating_regime", "error_by_operating_regime.png"), ("sequence_length_group", "error_by_sequence_length_group.png")]:
        plt.figure(figsize=(8, 5)); benchmark.groupby(column)["absolute_error"].mean().plot(kind="bar"); plt.xticks(rotation=30, ha="right"); save(fig_dir / name)
    transformer_screen = screening.assign(group=np.where(screening["architecture"].str.contains("transformer"), "transformer", screening["architecture"]))
    plt.figure(figsize=(8, 5)); transformer_screen.groupby("group")["validation_rmse"].mean().plot(kind="bar"); plt.xticks(rotation=30, ha="right"); save(fig_dir / "transformer_vs_lstm_comparison.png")
    plt.figure(figsize=(7, 5)); levels=[80,90,95]; plt.plot(levels,[uncertainty[f"covered_{p}"].mean() for p in levels],marker="o"); plt.plot(levels,[p/100 for p in levels],linestyle="--"); save(fig_dir / "coverage_vs_nominal_level.png")
    plt.figure(figsize=(8, 5)); uncertainty.groupby("subset")["interval_width_90"].mean().plot(kind="bar"); save(fig_dir / "interval_width_by_subset.png")
    plt.figure(figsize=(8, 5)); uncertainty_comparison_frame.pivot(index="subset", columns="model", values="mean_width_90").plot(kind="bar", ax=plt.gca()); save(fig_dir / "phase5_vs_phase5b_interval_width.png")
    plt.figure(figsize=(8, 5)); uncertainty.groupby("support_status")["abstain_flag"].mean().plot(kind="bar"); save(fig_dir / "abstention_tradeoff.png")
    plt.figure(figsize=(8, 5)); uncertainty["maintenance_action"].value_counts().plot(kind="bar"); plt.xticks(rotation=30, ha="right"); save(fig_dir / "maintenance_action_distribution.png")
    plt.figure(figsize=(8, 5)); benchmark.sort_values("absolute_error").head(20).plot(x="global_engine_id", y=["true_rul", "predicted_rul"], kind="bar", ax=plt.gca()); plt.xticks(rotation=90); save(fig_dir / "representative_engine_trajectories.png")
    examples = pd.concat([
        benchmark.nsmallest(2, "absolute_error"),
        benchmark.nlargest(2, "residual"),
        benchmark.nsmallest(2, "residual"),
        uncertainty.nsmallest(1, "interval_width_90"),
        uncertainty.nlargest(1, "interval_width_90"),
        uncertainty[uncertainty["abstain_flag"]].head(1),
        benchmark[benchmark["subset"] == "FD004"].head(1),
    ]).drop_duplicates("global_engine_id").head(10)
    for _, row in examples.iterrows():
        plt.figure(figsize=(6, 4)); plt.bar(["true", "pred"], [row["true_rul"], row["predicted_rul"]]); plt.title(str(row["global_engine_id"])); save(ex_dir / f"{row['global_engine_id']}_phase5b_example.png")
    return figures


def write_design_note(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Multidomain Temporal Optimization Design

Phase 5B preserves the frozen Phase 5 LSTM benchmark and evaluates whether bounded longer training, compact Transformer encoders, patch-based temporal attention, and longer-trained TCNs justify their extra cost. Four Phase 5 epochs are not treated as automatically insufficient: the locked epoch count is learned only from training-engine validation and may remain small when validation supports it. Longer training therefore uses early stopping, scheduler tracking, best-checkpoint restoration, and multi-seed validation rather than unconditional 50-epoch fitting.

The experiment uses train_FD001 through train_FD004 for model development and uses the corresponding test subsets only as held-out benchmark test sets. No benchmark labels select architecture, schedule, epoch count, uncertainty method, or abstention thresholds. Engine-group splits prevent windows from one engine appearing in both fitting and validation. Sequence windows remain past-only, use regime-standardized features, zero padding in standardized space, and a binary validity mask.

The fixed registry covers extended LSTM, extended causal TCN, compact cycle-token Transformer, and patch Transformer candidates. Full self-attention is allowed inside each past-only window because the target cycle is the final observation in the window; no observations after the target cycle are present. Padding masks and mask-aware pooling prevent padded tokens from influencing predictions. Attention weights, when inspected, are treated as diagnostics rather than complete causal explanations.

Finalist cross-validation and seed stability use training engines only. The locked model is refit on all four training subsets for the training-only locked epoch count, then evaluated once on the benchmark engines. Conformal intervals, support assessment, abstention, and demonstration maintenance recommendations reuse existing AeroGuard logic. The recommendations are not approved aircraft-maintenance instructions. C-MAPSS is a simulation benchmark, so results do not establish aircraft deployment readiness or certification.
""",
        encoding="utf-8",
        newline="\n",
    )


def write_results_note(path: Path, result: dict[str, Any], config_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Multidomain Temporal Optimization Results\n\n")
        for key in ["python_executable", "python_version", "torch_version", "cuda_available", "selected_execution_profile", "device", "training_engine_counts", "benchmark_engine_counts", "locked_architecture", "locked_schedule_id", "locked_epoch_count", "phase5b_point_conclusion", "runtime_by_stage"]:
            handle.write(f"- {key}: `{result.get(key)}`\n")
        handle.write(f"- Stage A results: `{result['stage_a_metrics']}`\n")
        handle.write(f"- Finalist CV results: `{result['finalist_cv_metrics']}`\n")
        handle.write(f"- Stability statistics: `{result['model_stability']}`\n")
        handle.write(f"- Benchmark metrics: `{result['benchmark_metrics']}`\n")
        handle.write(f"- Uncertainty method: `{result['locked_uncertainty_method']}`\n")
        handle.write(f"- Maintenance counts: `{result['maintenance_policy_metrics'].get('action_counts')}`\n")
        handle.write("\n## Generated Outputs\n\n")
        for item in result["generated_files"]:
            handle.write(f"- `{item}`\n")
        handle.write("\n## Warnings And Limitations\n\n")
        for warning in result["warnings"]:
            handle.write(f"- {warning}\n")
        handle.write("\n## Reproduction Command\n\n```powershell\n")
        handle.write('$env:PYTHONPATH = ".\\src"\n')
        handle.write("python -m aeroguard.pipelines.optimize_multidomain_temporal_rul ")
        handle.write(f'--config "{config_path.as_posix()}"\n')
        handle.write("```\n")


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    start = time.perf_counter()
    root = project_root()
    config_path = Path(config_path)
    config = load_config(config_path)
    output_dir = resolve_project_path(config["output_dir"], root)
    checkpoint_dir = resolve_project_path(config["checkpoint_dir"], root)
    for directory in [output_dir, output_dir / "figures", output_dir / "engine_examples", checkpoint_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    env = environment_report()
    env.update(
        {
            "sklearn_version": sklearn.__version__,
            "onnx_installed": importlib.util.find_spec("onnx") is not None,
            "onnxruntime_installed": importlib.util.find_spec("onnxruntime") is not None,
            "onnxscript_installed": importlib.util.find_spec("onnxscript") is not None,
        }
    )
    profile, limits = select_execution_profile(config, env)
    device = device_from_config(config, env)
    config = dict(config)
    config["pin_memory"] = bool(device.type == "cuda")
    set_global_seed(int(config["random_seed"]), bool(config.get("deterministic_algorithms", False)))
    print(f"Environment inspected: profile={profile}, device={device}")
    stage_times: dict[str, float] = {}
    manifest = create_phase5_manifest(output_dir, root, config)
    print("Phase 5 benchmark manifest created")

    data_start = time.perf_counter()
    dataset_dir = resolve_project_path(config["dataset_dir"], root)
    train_raw, train_meta = load_training_subsets(dataset_dir, config["training_subsets"], float(config["rul_cap"]), float(config["healthy_rul_threshold"]), float(config["critical_rul_threshold"]))
    train = assign_working_unit_ids(train_raw)
    test_raw, test_meta = load_test_subsets(dataset_dir, config["benchmark_test_subsets"], float(config["healthy_rul_threshold"]), float(config["critical_rul_threshold"]))
    test_frames = {subset: assign_working_unit_ids(frame) for subset, frame in test_raw.items()}
    stage_times["data_loading_seconds"] = time.perf_counter() - data_start
    print("Data loading complete")

    spec = WindowSpec(int(config["window_length"]), int(config["window_stride"]), int(config["minimum_valid_history"]))
    train_ids, val_ids = screening_split(train, float(config["screening_validation_fraction"]), int(config["screening_seed"]))
    write_json(output_dir / "screening_split.json", {"train_engine_ids": train_ids, "validation_engine_ids": val_ids, "engine_overlap": sorted(set(train_ids).intersection(val_ids))})
    pre = fit_preprocessor(train[train["global_engine_id"].isin(train_ids)].copy(), config)
    screen_train = apply_preprocessor(pre, train[train["global_engine_id"].isin(train_ids)].copy())
    screen_val = apply_preprocessor(pre, train[train["global_engine_id"].isin(val_ids)].copy())
    train_endpoints = build_endpoint_table(screen_train, spec, int(config["maximum_windows_per_engine"]), int(config["screening_seed"]))
    val_endpoints = snapshot_endpoint_table(screen_val, [float(x) for x in config["validation_snapshot_positions"]])
    train_dataset, _, train_sequences = make_dataset(screen_train, train_endpoints, pre["features"], spec)
    val_dataset, val_meta, _ = make_dataset(screen_val, val_endpoints, pre["features"], spec)
    sequence_audit(screen_train, train_endpoints, spec, config["rul_bands"]).to_csv(output_dir / "sequence_audit.csv", index=False)
    print("Window creation complete")

    registry = []
    screening_rows = []
    screening_models: dict[str, torch.nn.Module] = {}
    efficiency_rows = []
    example_single = torch.as_tensor(train_sequences[:1], dtype=torch.float32)
    example_batch = torch.as_tensor(train_sequences[: min(32, len(train_sequences))], dtype=torch.float32)
    input_dim = train_sequences.shape[2]
    for candidate in config["model_registry"]:
        candidate = dict(candidate)
        candidate["window_length"] = int(config["window_length"])
        schedule = schedule_for_candidate(config, candidate, limits, "stage_a")
        registry.append({"candidate": candidate, "schedule": schedule})
        model_id = str(candidate["model_id"])
        model, meta = train_candidate(candidate, train_dataset, val_dataset, val_meta, config, schedule, device, input_dim, int(config["random_seed"]) + len(screening_rows), bool(limits["effective_mixed_precision"]))
        pred = evaluate_model_frame(model, val_dataset, val_meta, device, int(config["batch_size"]), model_id)
        metrics = point_metrics_for_predictions(pred, float(config["abstention_settings"]["high_error_threshold"]))
        checkpoint_path = checkpoint_dir / f"stage_a_{model_id}.pt"
        save_checkpoint(checkpoint_path, model.to("cpu"), {"candidate": candidate, "schedule": schedule, **meta})
        row = {
            "model_id": model_id,
            "architecture": candidate["architecture"],
            "schedule_id": candidate["schedule_id"],
            "best_epoch": meta["best_epoch"],
            "stopping_epoch": meta["stopping_epoch"],
            "early_stopping_triggered": meta["early_stopping_triggered"],
            "parameter_count": trainable_parameter_count(model),
            "checkpoint_size_mb": checkpoint_size_mb(checkpoint_path),
            "training_runtime_seconds": meta["training_seconds"],
            "mean_epoch_runtime_seconds": meta["mean_epoch_seconds"],
            "validation_mae": metrics["mae"],
            "validation_rmse": metrics["rmse"],
            "validation_nasa_score": metrics["nasa_score"],
            "validation_optimistic_rate": metrics["optimistic_prediction_rate"],
            "validation_severe_optimistic_rate": metrics["severe_optimistic_error_rate"],
            "history": meta["history"],
        }
        screening_rows.append(row)
        screening_models[model_id] = deepcopy(model).to("cpu")
        eff = model_efficiency_row(model_id, model.to("cpu"), example_single, example_batch, device, meta, int(config.get("latency_repetitions", 100)))
        eff.update({"validation_rmse": metrics["rmse"], "architecture": candidate["architecture"], "schedule_id": candidate["schedule_id"]})
        if candidate["architecture"] == "patch_transformer" and hasattr(model, "patch_metadata"):
            eff.update(model.patch_metadata())
        elif candidate["architecture"] == "temporal_transformer":
            eff.update({"patch_token_count": int(config["window_length"]), "attention_layers": candidate["layers"], "attention_heads": candidate["heads"], "projection_dim": candidate["projection_dim"], "attention_complexity_scale": int(candidate["layers"] * candidate["heads"] * int(config["window_length"]) ** 2)})
        efficiency_rows.append(eff)
        print(f"Stage A candidate complete: {model_id} RMSE={metrics['rmse']:.3f}")
    write_json(output_dir / "extended_model_registry.json", {"registry": registry, "execution_profile": profile, "limits": limits})
    screening_df = pd.DataFrame(screening_rows).sort_values(["validation_rmse", "validation_nasa_score", "validation_optimistic_rate", "validation_mae", "parameter_count", "model_id"])
    screening_export = screening_df.copy()
    screening_export["history"] = screening_export["history"].map(json.dumps)
    screening_export.to_csv(output_dir / "screening_metrics.csv", index=False)
    efficiency = pd.DataFrame(efficiency_rows)

    finalist_ids = screening_df.head(int(limits["effective_finalist_count"]))["model_id"].tolist()
    print(f"Finalist selection complete: {finalist_ids}")
    splits = stratified_engine_group_splits(train, int(config["finalist_cv_folds"]), 1, [int(config["finalist_cv_seed"])])
    validate_no_engine_leakage(splits)
    cv_rows, cv_predictions = [], []
    finalist_seeds = [int(seed) for seed in config["finalist_seeds"][: int(limits["effective_finalist_seed_count"])]]
    for model_id in finalist_ids:
        candidate = next(dict(item) for item in config["model_registry"] if item["model_id"] == model_id)
        schedule = schedule_for_candidate(config, candidate, limits, "finalist")
        for fold_index, split in enumerate(splits, start=1):
            fold_pre = fit_preprocessor(train[train["global_engine_id"].isin(split.train_engine_ids)].copy(), config)
            fold_train = apply_preprocessor(fold_pre, train[train["global_engine_id"].isin(split.train_engine_ids)].copy())
            fold_val = apply_preprocessor(fold_pre, train[train["global_engine_id"].isin(split.validation_engine_ids)].copy())
            fold_train_endpoints = build_endpoint_table(fold_train, spec, int(config["maximum_windows_per_engine"]), int(config["finalist_cv_seed"]) + fold_index)
            fold_val_endpoints = snapshot_endpoint_table(fold_val, [float(x) for x in config["validation_snapshot_positions"]])
            fold_train_ds, _, fold_train_sequences = make_dataset(fold_train, fold_train_endpoints, fold_pre["features"], spec)
            fold_val_ds, fold_val_meta, _ = make_dataset(fold_val, fold_val_endpoints, fold_pre["features"], spec)
            for seed in finalist_seeds:
                model, meta = train_candidate(candidate, fold_train_ds, fold_val_ds, fold_val_meta, config, schedule, device, fold_train_sequences.shape[2], seed, bool(limits["effective_mixed_precision"]))
                pred = evaluate_model_frame(model, fold_val_ds, fold_val_meta, device, int(config["batch_size"]), model_id)
                pred["fold"] = split.split_id
                pred["seed"] = seed
                cv_predictions.append(pred)
                metrics = point_metrics_for_predictions(pred, float(config["abstention_settings"]["high_error_threshold"]))
                cv_rows.append({"model_id": model_id, "architecture": candidate["architecture"], "schedule_id": candidate["schedule_id"], "fold": split.split_id, "seed": seed, "fitting_engine_count": len(split.train_engine_ids), "validation_engine_count": len(split.validation_engine_ids), "best_epoch": meta["best_epoch"], "stopping_epoch": meta["stopping_epoch"], "parameter_count": trainable_parameter_count(model), "runtime_seconds": meta["training_seconds"], "validation_mae": metrics["mae"], "validation_rmse": metrics["rmse"], "validation_nasa_score": metrics["nasa_score"], "validation_optimistic_rate": metrics["optimistic_prediction_rate"], "validation_severe_optimistic_rate": metrics["severe_optimistic_error_rate"]})
                save_checkpoint(checkpoint_dir / f"finalist_{model_id}_{split.split_id}_seed{seed}.pt", model.to("cpu"), {"candidate": candidate, "schedule": schedule, "fold": split.to_dict(), "seed": seed, **meta})
                print(f"Finalist fold/seed complete: {model_id} {split.split_id} seed={seed} RMSE={metrics['rmse']:.3f}")
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(output_dir / "finalist_cross_validation_metrics.csv", index=False)
    oof = pd.concat(cv_predictions, ignore_index=True)
    aggregate_seed_metrics(cv_df).to_csv(output_dir / "seed_summary.csv", index=False)
    prediction_disagreement(oof).to_csv(output_dir / "seed_prediction_disagreement.csv", index=False)
    stability = summarize_model_stability(cv_df)
    stability.to_csv(output_dir / "model_stability.csv", index=False)
    ranking = stability.sort_values(["mean_rmse", "std_rmse", "mean_nasa_score", "mean_optimistic_rate", "model_id"]).copy()
    ranking.to_csv(output_dir / "extended_model_ranking.csv", index=False)
    locked_model_id = str(ranking.iloc[0]["model_id"])
    locked_candidate = next(dict(item) for item in config["model_registry"] if item["model_id"] == locked_model_id)
    epoch_info = locked_epoch_from_cv(cv_df, locked_model_id, int(config["training_schedules"][locked_candidate["schedule_id"]]["max_epochs"]))
    locked_epoch_count = int(epoch_info["locked_epoch_count"])
    locked_schedule = schedule_for_candidate(config, locked_candidate, limits, "finalist")
    locked_info = {**locked_candidate, **epoch_info, "selection_source": "training-engine CV and seed stability only", "benchmark_tests_used_for_selection": False}
    print(f"Locked model selection complete: {locked_model_id}, epochs={locked_epoch_count}")

    final_start = time.perf_counter()
    final_pre = fit_preprocessor(train.copy(), config)
    final_train = apply_preprocessor(final_pre, train.copy())
    final_endpoints = build_endpoint_table(final_train, spec, int(config["maximum_windows_per_engine"]), int(config["random_seed"]))
    final_ds, _, final_sequences = make_dataset(final_train, final_endpoints, final_pre["features"], spec)
    final_model, final_meta = train_locked(locked_candidate, final_ds, config, locked_schedule, device, final_sequences.shape[2], int(config["random_seed"]), locked_epoch_count, bool(limits["effective_mixed_precision"]))
    locked_checkpoint = checkpoint_dir / "locked_extended_model.pt"
    save_checkpoint(locked_checkpoint, final_model.to("cpu"), {"candidate": locked_candidate, "schedule": locked_schedule, "locked_epoch_count": locked_epoch_count, **final_meta})
    final_model.to(device)
    stage_times["final_training_seconds"] = time.perf_counter() - final_start
    locked_info.update({"locked_epoch_count": locked_epoch_count, "parameter_count": trainable_parameter_count(final_model), "checkpoint_size_mb": checkpoint_size_mb(locked_checkpoint)})
    write_json(output_dir / "locked_extended_model.json", locked_info)
    print("Final fitting complete")

    bench_rows, transformed_final_rows = [], []
    for subset, frame in test_frames.items():
        transformed = apply_preprocessor(final_pre, frame.copy())
        if config["training_target"] not in transformed.columns:
            transformed[config["training_target"]] = transformed["true_rul_uncapped"].clip(upper=float(config["rul_cap"]))
        endpoints = final_endpoint_table(transformed)
        ds, meta, _ = make_dataset(transformed, endpoints, final_pre["features"], spec)
        pred = evaluate_model_frame(final_model, ds, meta, device, int(config["batch_size"]), locked_model_id)
        pred["subset"] = subset
        pred["final_observed_cycle"] = pred["cycle"]
        bench_rows.append(pred)
        transformed_final_rows.append(transformed.sort_values(["global_engine_id", CYCLE_COLUMN]).groupby("global_engine_id").tail(1).copy())
    benchmark = pd.concat(bench_rows, ignore_index=True)
    transformed_final = pd.concat(transformed_final_rows, ignore_index=True)
    benchmark["sequence_length_group"] = pd.cut(benchmark["sequence_valid_length"], bins=[0, 20, 40, 1000], labels=["short", "medium", "long"], include_lowest=True).astype(str)
    benchmark["true_rul_band"] = assign_numeric_band(benchmark["true_rul"], config["rul_bands"], "true_rul_band")
    benchmark.to_csv(output_dir / "benchmark_predictions.csv", index=False)
    print("Benchmark evaluation complete")

    metrics_by_subset = metrics_by_group(benchmark, "subset", float(config["abstention_settings"]["high_error_threshold"]))
    metrics_by_rul = metrics_by_group(benchmark, "true_rul_band", float(config["abstention_settings"]["high_error_threshold"]))
    metrics_by_regime = metrics_by_group(benchmark, "operating_regime", float(config["abstention_settings"]["high_error_threshold"]))
    metrics_by_subset.to_csv(output_dir / "metrics_by_subset.csv", index=False)
    metrics_by_rul.to_csv(output_dir / "metrics_by_rul_band.csv", index=False)
    metrics_by_regime.to_csv(output_dir / "metrics_by_regime.csv", index=False)
    benchmark_metrics = {subset: point_metrics_for_predictions(group, float(config["abstention_settings"]["high_error_threshold"])) for subset, group in benchmark.groupby("subset")}
    benchmark_metrics["overall"] = point_metrics_for_predictions(benchmark, float(config["abstention_settings"]["high_error_threshold"]))
    write_json(output_dir / "benchmark_metrics.json", benchmark_metrics)

    locked_oof = oof[oof["model_id"] == locked_model_id].copy()
    locked_uncertainty, unc_cv_metrics, calibrators = choose_uncertainty_method(locked_oof, config)
    unc_cv_metrics.to_csv(output_dir / "uncertainty_cv_metrics.csv", index=False)
    write_json(output_dir / "locked_uncertainty_method.json", locked_uncertainty)
    cal = calibrators["band"] if calibrators["locked_method_id"] == "phase5b_predicted_band_conformal" else calibrators["global"]
    uncertainty = add_uncertainty(benchmark, cal, calibrators["locked_method_id"], [float(x) for x in config["nominal_coverage_levels"]])
    median_width90 = float(add_uncertainty(locked_oof, cal, calibrators["locked_method_id"], [float(x) for x in config["nominal_coverage_levels"]])["interval_width_90"].median())
    uncertainty = apply_support_abstention_maintenance(uncertainty, transformed_final, final_pre, final_train, config, median_width90)
    uncertainty.to_csv(output_dir / "uncertainty_predictions.csv", index=False)
    uncertainty_metrics = {
        subset: {str(level): interval_metrics(group["true_rul"], group["predicted_rul"], group[f"lower_{int(round(level * 100))}"], group[f"upper_{int(round(level * 100))}"], level) for level in config["nominal_coverage_levels"]}
        for subset, group in uncertainty.groupby("subset")
    }
    uncertainty_metrics["overall"] = {str(level): interval_metrics(uncertainty["true_rul"], uncertainty["predicted_rul"], uncertainty[f"lower_{int(round(level * 100))}"], uncertainty[f"upper_{int(round(level * 100))}"], level) for level in config["nominal_coverage_levels"]}
    write_json(output_dir / "uncertainty_metrics.json", uncertainty_metrics)
    abst_metrics = {subset: abstention_metrics(group, 90, float(config["abstention_settings"]["high_error_threshold"])) for subset, group in uncertainty.groupby("subset")}
    write_json(output_dir / "abstention_metrics.json", abst_metrics)
    maintenance_recs = uncertainty[["subset", "global_engine_id", UNIT_COLUMN, "true_rul", "predicted_rul", "lower_90", "upper_90", "support_status", "support_score", "feature_exceedance_fraction", "regime_distance", "interval_width_ratio", "abstain_flag", "abstention_reason", "maintenance_action", "action_basis", "conservative_rul_bound", "nominal_interval_level", "prediction_status", "maintenance_disclaimer"]]
    maintenance_recs.to_csv(output_dir / "maintenance_recommendations.csv", index=False)
    maintenance_metrics = maintenance_policy_metrics(uncertainty)
    write_json(output_dir / "maintenance_policy_metrics.json", maintenance_metrics)
    print("Conformal calibration complete")

    phase5_comparison, point_conclusion = compare_phase5_phase5b(output_dir, root, config, benchmark_metrics, locked_info, final_meta, efficiency)
    unc_comparison = uncertainty_comparison(output_dir, root, config, uncertainty, uncertainty_metrics)
    locked_eff = model_efficiency_row(locked_model_id, final_model.to("cpu"), example_single, example_batch, device, final_meta, int(config.get("latency_repetitions", 100)))
    locked_eff.update({"model_id": locked_model_id, "architecture": locked_candidate["architecture"], "schedule_id": locked_candidate["schedule_id"], "validation_rmse": float(ranking.iloc[0]["mean_rmse"])})
    efficiency = pd.concat([efficiency, pd.DataFrame([locked_eff])], ignore_index=True)
    efficiency.to_csv(output_dir / "model_efficiency.csv", index=False)

    onnx_report = {"onnx_installed": True, "onnxruntime_installed": True, "onnxscript_installed": False, "exported": False, "reason": "onnxscript is not installed; package installation is prohibited."}
    figures = make_figures(output_dir, screening_df, cv_df, stability, benchmark, phase5_comparison, uncertainty, unc_comparison, efficiency)
    print("Figure generation complete")
    final_metadata = {
        "preprocessor": {"features": final_pre["features"], "retained_raw_features": final_pre["retained"], "excluded_features": final_pre["excluded"], "normalization": final_pre["normalizer"].metadata()},
        "locked_model": locked_candidate,
        "locked_schedule": locked_schedule,
        "locked_epoch_count": locked_epoch_count,
        "support": uncertainty.attrs.get("support_metadata", {}),
        "software": env,
        "onnx": onnx_report,
    }
    write_json(output_dir / "final_fit_metadata.json", final_metadata)
    design_note = root / "notes" / "multidomain_temporal_optimization_design.md"
    results_note = root / "notes" / "multidomain_temporal_optimization_results.md"
    write_design_note(design_note)
    stage_times["total_runtime_seconds"] = time.perf_counter() - start
    generated_files = [str(path) for path in [
        output_dir / "phase5_benchmark_manifest.json", output_dir / "extended_model_registry.json", output_dir / "screening_split.json", output_dir / "screening_metrics.csv", output_dir / "finalist_cross_validation_metrics.csv", output_dir / "model_stability.csv", output_dir / "extended_model_ranking.csv", output_dir / "locked_extended_model.json", output_dir / "final_fit_metadata.json", output_dir / "benchmark_predictions.csv", output_dir / "benchmark_metrics.json", output_dir / "metrics_by_subset.csv", output_dir / "metrics_by_rul_band.csv", output_dir / "metrics_by_regime.csv", output_dir / "phase5_vs_phase5b.csv", output_dir / "uncertainty_cv_metrics.csv", output_dir / "locked_uncertainty_method.json", output_dir / "uncertainty_predictions.csv", output_dir / "uncertainty_metrics.json", output_dir / "phase5_vs_phase5b_uncertainty.csv", output_dir / "model_efficiency.csv", output_dir / "abstention_metrics.json", output_dir / "maintenance_recommendations.csv", output_dir / "maintenance_policy_metrics.json", output_dir / "run_summary.json", output_dir / "seed_summary.csv", output_dir / "seed_prediction_disagreement.csv"]]
    generated_files.extend(figures)
    generated_files.extend(str(path) for path in checkpoint_dir.glob("*"))
    generated_files.extend([str(design_note), str(results_note)])
    result = {
        **env,
        "selected_execution_profile": profile,
        "execution_profile_reason": limits["reason"],
        "device": str(device),
        "runtime_by_stage": stage_times,
        "training_metadata": train_meta,
        "benchmark_test_metadata": test_meta,
        "training_engine_counts": train.groupby("source_domain")["global_engine_id"].nunique().to_dict(),
        "benchmark_engine_counts": {subset: int(frame["global_engine_id"].nunique()) for subset, frame in test_frames.items()},
        "sequence_settings": {"window_length": spec.window_length, "window_stride": spec.stride, "minimum_valid_history": spec.minimum_valid_history, "maximum_windows_per_engine": int(config["maximum_windows_per_engine"]), "stage_a_window_count": int(len(train_dataset)), "validation_snapshot_count": int(len(val_dataset)), "final_training_window_count": int(len(final_ds))},
        "candidate_registry": registry,
        "training_schedules": config["training_schedules"],
        "stage_a_metrics": screening_df.drop(columns=["history"]).to_dict(orient="records"),
        "best_epoch_by_candidate": screening_df[["model_id", "best_epoch", "stopping_epoch"]].to_dict(orient="records"),
        "finalists": finalist_ids,
        "finalist_cv_metrics": cv_df.to_dict(orient="records"),
        "seed_summary": aggregate_seed_metrics(cv_df).to_dict(orient="records"),
        "model_stability": stability.to_dict(orient="records"),
        "locked_architecture": locked_candidate["architecture"],
        "locked_model_id": locked_model_id,
        "locked_schedule_id": locked_candidate["schedule_id"],
        "locked_epoch_count": locked_epoch_count,
        "selection_rationale": ranking.iloc[0].to_dict(),
        "benchmark_metrics": benchmark_metrics,
        "phase5_vs_phase5b": phase5_comparison.to_dict(orient="records"),
        "phase5b_point_conclusion": point_conclusion,
        "locked_uncertainty_method": locked_uncertainty,
        "uncertainty_metrics": uncertainty_metrics,
        "phase5_vs_phase5b_uncertainty": unc_comparison.to_dict(orient="records"),
        "abstention_metrics": abst_metrics,
        "maintenance_policy_metrics": maintenance_metrics,
        "model_efficiency": efficiency.to_dict(orient="records"),
        "phase5_benchmark_manifest": manifest,
        "generated_files": generated_files,
        "warnings": [
            "Benchmark test subsets were not used for model or schedule selection.",
            "Automatic execution limits may reduce finalist count, seed count, and effective epoch caps for local hardware safety.",
            "ONNX export skipped because onnxscript is not installed and package installation is prohibited.",
            "Maintenance recommendations are demonstration decision-support outputs, not approved aircraft-maintenance instructions.",
        ],
    }
    write_results_note(results_note, result, config_path)
    write_json(output_dir / "run_summary.json", result)
    print(json.dumps(_json_ready(result), indent=2, allow_nan=False))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 5B multidomain temporal RUL optimization.")
    parser.add_argument("--config", required=True, help="Path to Phase 5B YAML config.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
