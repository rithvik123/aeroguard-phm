"""Train and evaluate multidomain temporal deep RUL models."""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.preprocessing import StandardScaler

from aeroguard.data.columns import CYCLE_COLUMN, UNIT_COLUMN
from aeroguard.data.multi_subset import load_test_subsets, load_training_subsets
from aeroguard.deep.checkpoints import save_checkpoint
from aeroguard.deep.inference import predict_batches
from aeroguard.deep.models import MODEL_CLASSES
from aeroguard.deep.models.common import trainable_parameter_count, validate_parameter_budget
from aeroguard.deep.reproducibility import set_global_seed
from aeroguard.deep.sampling import build_endpoint_table
from aeroguard.deep.sequence_dataset import InferenceSequenceDataset, SequenceWindowDataset
from aeroguard.deep.training import train_fixed_epochs, train_model
from aeroguard.deep.windowing import WindowSpec, build_inference_windows, build_training_windows, endpoints_for_normalized_positions, final_endpoint_table, sequence_audit
from aeroguard.evaluation.coverage_analysis import assign_numeric_band, coverage_by_group
from aeroguard.evaluation.deep_rul_metrics import deep_point_metrics, metrics_by_group, prediction_direction
from aeroguard.evaluation.leave_one_domain_out import stratified_engine_group_splits, validate_no_engine_leakage
from aeroguard.evaluation.model_efficiency import model_efficiency_row
from aeroguard.evaluation.uncertainty_metrics import interval_metrics
from aeroguard.features.condition_normalization import ConditionNormalizer
from aeroguard.maintenance.uncertainty_policy import assign_maintenance_recommendations, maintenance_policy_metrics
from aeroguard.pipelines.train_fd001_baseline import project_root, resolve_project_path
from aeroguard.pipelines.train_fd001_health_anomaly import select_anomaly_features
from aeroguard.pipelines.train_multidomain_phm import assign_working_unit_ids
from aeroguard.uncertainty.abstention import abstention_metrics, apply_abstention
from aeroguard.uncertainty.conformal import GlobalConformalCalibrator, PredictedRulBandConformalCalibrator
from aeroguard.uncertainty.support import SupportModel


REQUIRED_CONFIG_KEYS = {
    "dataset_dir",
    "training_subsets",
    "benchmark_test_subsets",
    "phase3_config_path",
    "phase4_config_path",
    "phase3_results_path",
    "phase4_results_path",
    "random_seed",
    "execution_profile",
    "device",
    "window_length",
    "window_stride",
    "minimum_valid_history",
    "maximum_windows_per_engine",
    "rul_bands",
    "training_target",
    "rul_cap",
    "include_cycle_as_feature",
    "operating_condition_method",
    "number_of_operating_regimes",
    "screening_validation_fraction",
    "screening_seed",
    "finalist_count",
    "finalist_cv_folds",
    "finalist_cv_seed",
    "validation_snapshot_positions",
    "model_registry",
    "parameter_budget",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "loss",
    "gradient_clip_norm",
    "screening_max_epochs",
    "finalist_max_epochs",
    "early_stopping_patience",
    "batch_size",
    "num_workers",
    "mixed_precision",
    "deep_improvement_classification",
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


PROFILE_LIMITS = {
    "gpu_standard": {
        "screening_max_epochs": 25,
        "early_stopping_patience": 5,
        "batch_size": 256,
        "finalist_max_epochs": 20,
        "finalist_count": 3,
        "finalist_cv_folds": 3,
        "mixed_precision": True,
        "parameter_budget": 2_000_000,
    },
    "cpu_safe": {
        "screening_max_epochs": 10,
        "early_stopping_patience": 3,
        "batch_size": 128,
        "finalist_max_epochs": 8,
        "finalist_count": 2,
        "finalist_cv_folds": 3,
        "mixed_precision": False,
        "parameter_budget": 1_000_000,
    },
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


def environment_report() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    memory = None
    gpu_names = []
    if cuda:
        for index in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_name(index))
        try:
            free, total = torch.cuda.mem_get_info(0)
            memory = {"free_bytes": int(free), "total_bytes": int(total)}
        except Exception as exc:  # pragma: no cover - hardware dependent
            memory = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": bool(cuda),
        "cuda_version": torch.version.cuda,
        "gpu_names": gpu_names,
        "visible_gpu_count": int(torch.cuda.device_count()),
        "gpu_memory": memory,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "mixed_precision_available": bool(hasattr(torch, "amp") and hasattr(torch.cuda, "amp")),
    }


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_config(config, project_root())
    return config


def _validate_bands(bands: list[dict[str, Any]]) -> None:
    previous = -math.inf
    for band in bands:
        lower = float(band["lower"])
        upper = band.get("upper")
        upper_value = math.inf if upper is None else float(upper)
        if lower < previous or upper_value < lower:
            raise ValueError("Invalid RUL band ordering.")
        previous = upper_value


def validate_config(config: dict[str, Any], root: Path) -> None:
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"Missing required configuration keys: {missing}")
    dataset_dir = resolve_project_path(config["dataset_dir"], root)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    for subset in [str(item).upper() for item in config["training_subsets"]]:
        if not (dataset_dir / f"train_{subset}.txt").exists():
            raise FileNotFoundError(f"Missing training file for {subset}.")
    for subset in [str(item).upper() for item in config["benchmark_test_subsets"]]:
        for filename in [f"test_{subset}.txt", f"RUL_{subset}.txt"]:
            if not (dataset_dir / filename).exists():
                raise FileNotFoundError(f"Missing benchmark file: {filename}")
    for key in ["phase3_config_path", "phase4_config_path", "phase3_results_path", "phase4_results_path"]:
        if not resolve_project_path(config[key], root).exists():
            raise FileNotFoundError(f"Missing benchmark path: {config[key]}")
    if int(config["window_length"]) <= 0 or int(config["window_stride"]) <= 0:
        raise ValueError("Invalid window length or stride.")
    if not 1 <= int(config["minimum_valid_history"]) <= int(config["window_length"]):
        raise ValueError("Invalid minimum_valid_history.")
    if int(config["maximum_windows_per_engine"]) <= 0:
        raise ValueError("Invalid maximum_windows_per_engine.")
    _validate_bands(config["rul_bands"])
    _validate_bands(config["predicted_rul_bands"])
    ids = [str(item["model_id"]) for item in config["model_registry"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate model IDs.")
    unsupported = [item["architecture"] for item in config["model_registry"] if item["architecture"] not in MODEL_CLASSES]
    if unsupported:
        raise ValueError(f"Unsupported architectures: {unsupported}")
    if not 0.0 < float(config["screening_validation_fraction"]) < 1.0:
        raise ValueError("Invalid screening validation fraction.")
    if int(config["finalist_cv_folds"]) < 2 or int(config["finalist_count"]) < 1:
        raise ValueError("Invalid finalist CV settings.")
    if any(not 0.0 < float(level) < 1.0 for level in config["nominal_coverage_levels"]):
        raise ValueError("Invalid nominal coverage levels.")
    if float(config["learning_rate"]) <= 0 or int(config["batch_size"]) <= 0:
        raise ValueError("Invalid optimizer settings.")
    output_dir = resolve_project_path(config["output_dir"], root)
    checkpoint_dir = resolve_project_path(config["checkpoint_dir"], root)
    design_note_path = resolve_project_path(config.get("design_note_path", "notes/multidomain_deep_rul_design.md"), root)
    results_note_path = resolve_project_path(config.get("results_note_path", "notes/multidomain_deep_rul_results.md"), root)
    for path in [output_dir, checkpoint_dir, design_note_path, results_note_path]:
        lowered = str(path).lower()
        if "\\references\\" in lowered or "\\extracted-code\\" in lowered:
            raise ValueError("Outputs must not be inside protected directories.")


def select_execution_profile(config: dict[str, Any], env: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    requested = str(config["execution_profile"])
    if requested == "auto":
        profile = "gpu_standard" if bool(env["cuda_available"]) else "cpu_safe"
        reason = "CUDA is available." if profile == "gpu_standard" else "CUDA is unavailable."
    else:
        profile = requested
        reason = "Profile explicitly configured."
    if profile not in PROFILE_LIMITS:
        raise ValueError(f"Unsupported execution profile: {profile}")
    limits = dict(PROFILE_LIMITS[profile])
    limits["reason"] = reason
    limits["effective_screening_max_epochs"] = min(int(config["screening_max_epochs"]), int(limits["screening_max_epochs"]))
    limits["effective_finalist_max_epochs"] = min(int(config["finalist_max_epochs"]), int(limits["finalist_max_epochs"]))
    limits["effective_patience"] = min(int(config["early_stopping_patience"]), int(limits["early_stopping_patience"]))
    limits["effective_batch_size"] = min(int(config["batch_size"]), int(limits["batch_size"]))
    limits["effective_finalist_count"] = min(int(config["finalist_count"]), int(limits["finalist_count"]))
    limits["effective_finalist_cv_folds"] = min(int(config["finalist_cv_folds"]), int(limits["finalist_cv_folds"]))
    limits["effective_mixed_precision"] = bool(limits["mixed_precision"] and config["mixed_precision"] != "false" and env["cuda_available"])
    limits["effective_parameter_budget"] = min(int(config["parameter_budget"][profile]), int(limits["parameter_budget"]))
    return profile, limits


def device_from_config(config: dict[str, Any], env: dict[str, Any]) -> torch.device:
    requested = str(config["device"])
    if requested == "auto":
        return torch.device("cuda" if env["cuda_available"] else "cpu")
    return torch.device(requested)


def create_classical_manifest(output_dir: Path, root: Path, config: dict[str, Any]) -> dict[str, Any]:
    phase3 = resolve_project_path(config["phase3_results_path"], root)
    phase4 = resolve_project_path(config["phase4_results_path"], root)
    files = [
        phase3 / "run_summary.json",
        phase3 / "fd004_external_metrics.json",
        phase3 / "rul_transfer_metrics.json",
        phase4 / "run_summary.json",
        phase4 / "uncertainty_predictions.csv",
        phase4 / "calibration_conclusion.json",
        phase4 / "bootstrap_confidence_intervals.json",
    ]
    hashes = {str(path): sha256_file(path) for path in files if path.exists()}
    phase4_summary = json.loads((phase4 / "run_summary.json").read_text(encoding="utf-8"))
    manifest = {
        "phase3_results_path": str(phase3),
        "phase4_results_path": str(phase4),
        "sha256": hashes,
        "classical_point_model": phase4_summary["locked_point_model"],
        "classical_uncertainty_method": phase4_summary["locked_uncertainty_method"]["method_id"],
        "metrics_by_subset": phase4_summary["test_metrics_by_subset"],
        "fd004_mae": phase4_summary["test_metrics_by_subset"]["FD004"]["point"]["mae"],
        "fd004_rmse": phase4_summary["test_metrics_by_subset"]["FD004"]["point"]["rmse"],
        "fd004_90_coverage": phase4_summary["test_metrics_by_subset"]["FD004"]["intervals"]["0.9"]["coverage"],
        "fd004_90_interval_width": phase4_summary["test_metrics_by_subset"]["FD004"]["intervals"]["0.9"]["mean_interval_width"],
        "fd004_abstention_rate": phase4_summary["abstention_metrics"]["FD004"]["abstention_rate"],
        "statement": "Previous Phase 3 and Phase 4 benchmark files were read only and not modified.",
    }
    write_json(output_dir / "classical_benchmark_manifest.json", manifest)
    return manifest


def fit_preprocessor(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    _, retained, reasons, _, _ = select_anomaly_features(
        frame,
        include_cycle=bool(config["include_cycle_as_feature"]),
        configured_exclusions=list(config.get("features_to_exclude", [])),
        near_constant_threshold=float(config["near_constant_threshold"]),
        correlation_threshold=float(config["correlation_threshold"]),
    )
    normalizer = ConditionNormalizer(
        method=str(config["operating_condition_method"]),
        n_regimes=int(config["number_of_operating_regimes"]),
        random_state=int(config["random_seed"]),
        ridge_alpha=float(config["residualization_ridge_alpha"]),
    ).fit(frame, retained)
    transformed = normalizer.transform(frame)
    features = normalizer.output_features_
    scaler = StandardScaler().fit(transformed[features])
    return {"normalizer": normalizer, "scaler": scaler, "features": features, "retained": retained, "excluded": reasons}


def apply_preprocessor(preprocessor: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    transformed = preprocessor["normalizer"].transform(frame)
    features = preprocessor["features"]
    transformed.loc[:, features] = preprocessor["scaler"].transform(transformed[features])
    return transformed


def screening_split(frame: pd.DataFrame, fraction: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(int(seed))
    train_ids, validation_ids = [], []
    domains = frame[["source_domain", "global_engine_id"]].drop_duplicates()
    for _, group in domains.groupby("source_domain"):
        ids = np.asarray(sorted(group["global_engine_id"].tolist()), dtype=object)
        rng.shuffle(ids)
        val_count = max(1, int(round(len(ids) * float(fraction))))
        validation_ids.extend(str(item) for item in ids[:val_count])
        train_ids.extend(str(item) for item in ids[val_count:])
    return sorted(train_ids), sorted(validation_ids)


def snapshot_endpoint_table(frame: pd.DataFrame, positions: list[float]) -> pd.DataFrame:
    rows = []
    for engine, group in frame.sort_values(["global_engine_id", CYCLE_COLUMN]).groupby("global_engine_id"):
        endpoints = endpoints_for_normalized_positions(group.reset_index(drop=True), positions)
        for endpoint in endpoints:
            rows.append({"global_engine_id": engine, "endpoint_index": int(endpoint)})
    return pd.DataFrame(rows)


def make_dataset(
    frame: pd.DataFrame,
    endpoint_table: pd.DataFrame,
    features: list[str],
    spec: WindowSpec,
    *,
    mode: str = "training",
) -> tuple[SequenceWindowDataset | InferenceSequenceDataset, pd.DataFrame, np.ndarray]:
    if mode == "training":
        sequences, metadata = build_training_windows(frame, features, endpoint_table, spec)
        dataset = SequenceWindowDataset(sequences, metadata["target_rul_capped"].to_numpy(dtype=np.float32), metadata["sequence_valid_length"].to_numpy(dtype=np.int64))
    elif mode == "inference":
        sequences, metadata = build_inference_windows(frame, features, endpoint_table, spec)
        dataset = InferenceSequenceDataset(sequences, metadata["sequence_valid_length"].to_numpy(dtype=np.int64))
    else:
        raise ValueError("mode must be 'training' or 'inference'.")
    return dataset, metadata, sequences


def build_model(model_config: dict[str, Any], input_dim: int, budget: int) -> torch.nn.Module:
    architecture = str(model_config["architecture"])
    params = {key: value for key, value in model_config.items() if key not in {"model_id", "architecture"}}
    model = MODEL_CLASSES[architecture](input_dim=input_dim, **params)
    validate_parameter_budget(model, budget)
    return model


def evaluate_model_frame(model: torch.nn.Module, dataset: SequenceWindowDataset, metadata: pd.DataFrame, device: torch.device, batch_size: int, model_id: str) -> pd.DataFrame:
    pred = predict_batches(model, dataset, device, batch_size=batch_size)
    result = metadata.copy()
    result["model_id"] = model_id
    result["predicted_rul_raw"] = pred
    result["predicted_rul"] = np.maximum(0.0, pred)
    result["true_rul"] = result["target_rul_uncapped"].astype(float)
    result["residual"] = result["predicted_rul"] - result["true_rul"]
    result["absolute_error"] = result["residual"].abs()
    result["squared_error"] = np.square(result["residual"])
    result["prediction_direction"] = [prediction_direction(value) for value in result["residual"]]
    return result


def train_one_model(
    model_config: dict[str, Any],
    train_dataset: SequenceWindowDataset,
    validation_dataset: SequenceWindowDataset,
    train_config: dict[str, Any],
    device: torch.device,
    budget: int,
    input_dim: int,
    max_epochs: int,
    patience: int,
    mixed_precision: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    set_global_seed(int(train_config["random_seed"]), bool(train_config.get("deterministic_algorithms", False)))
    model = build_model(model_config, input_dim, budget)
    try:
        return train_model(model, train_dataset, validation_dataset, train_config, device, max_epochs, patience, mixed_precision)
    except RuntimeError as exc:
        if not mixed_precision or "Non-finite gradients" not in str(exc):
            raise
        set_global_seed(int(train_config["random_seed"]), bool(train_config.get("deterministic_algorithms", False)))
        model = build_model(model_config, input_dim, budget)
        model, metadata = train_model(model, train_dataset, validation_dataset, train_config, device, max_epochs, patience, False)
        metadata["mixed_precision_retry_reason"] = "disabled_after_non_finite_gradients"
        return model, metadata


def point_metrics_for_predictions(frame: pd.DataFrame, severe_threshold: float = 30.0) -> dict[str, Any]:
    return deep_point_metrics(frame["true_rul"], frame["predicted_rul"], severe_threshold)


def add_uncertainty(predictions: pd.DataFrame, calibrator: Any, method_id: str, levels: list[float], band: bool = False) -> pd.DataFrame:
    intervals = calibrator.interval_frame(predictions["predicted_rul"])
    result = pd.concat([predictions.reset_index(drop=True), intervals.reset_index(drop=True)], axis=1)
    result["uncertainty_method_id"] = method_id
    for level in levels:
        pct = int(round(level * 100))
        result[f"interval_width_{pct}"] = result[f"upper_{pct}"] - result[f"lower_{pct}"]
        result[f"covered_{pct}"] = (result["true_rul"] >= result[f"lower_{pct}"]) & (result["true_rul"] <= result[f"upper_{pct}"])
    return result


def uncertainty_metrics(frame: pd.DataFrame, method_id: str, levels: list[float]) -> list[dict[str, Any]]:
    rows = []
    for level in levels:
        pct = int(round(level * 100))
        rows.append({"uncertainty_method_id": method_id, **interval_metrics(frame["true_rul"], frame["predicted_rul"], frame[f"lower_{pct}"], frame[f"upper_{pct}"], level)})
    return rows


def choose_uncertainty_method(oof: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    levels = [float(level) for level in config["nominal_coverage_levels"]]
    global_cal = GlobalConformalCalibrator(levels).fit(oof["residual"])
    band_cal = PredictedRulBandConformalCalibrator(levels, config["predicted_rul_bands"], minimum_samples_per_band=30).fit(oof["predicted_rul"], oof["residual"])
    candidates = {
        "deep_global_grouped_conformal": global_cal,
        "deep_predicted_band_conformal": band_cal,
    }
    rows = []
    frames = {}
    for method_id, calibrator in candidates.items():
        frame = add_uncertainty(oof, calibrator, method_id, levels)
        frames[method_id] = frame
        rows.extend(uncertainty_metrics(frame, method_id, levels))
    metrics = pd.DataFrame(rows)
    selected90 = metrics[metrics["nominal_level"] == 0.90].copy()
    selected90["feasible"] = selected90["coverage"] >= 0.90 - float(config["coverage_tolerance"])
    selected90["selection_score"] = np.where(selected90["feasible"], 0.0, 10.0) + selected90["undercoverage_amount"] * 20.0 + selected90["mean_interval_width"] / 100.0 + selected90["winkler_interval_score"] / 1000.0
    ranking = selected90.sort_values(["feasible", "selection_score", "mean_interval_width"], ascending=[False, True, True])
    locked = str(ranking.iloc[0]["uncertainty_method_id"])
    return {"method_id": locked, "calibrator": candidates[locked].metadata(), "fd004_used_for_selection": False}, metrics, {"global": global_cal, "band": band_cal, "locked_method_id": locked}


def apply_support_abstention_maintenance(predictions: pd.DataFrame, transformed_final: pd.DataFrame, preprocessor: dict[str, Any], train_transformed: pd.DataFrame, config: dict[str, Any], median_width90: float) -> pd.DataFrame:
    support_cfg = config["support_settings"]
    low, high = [float(item) for item in support_cfg["support_percentile_range"]]
    support_model = SupportModel(
        feature_columns=preprocessor["features"],
        percentile_low=low,
        percentile_high=high,
        feature_exceedance_limited=float(support_cfg["limited_feature_exceedance"]),
        feature_exceedance_out=float(support_cfg["out_feature_exceedance"]),
        robust_distance_limited=float(support_cfg["limited_robust_distance"]),
        robust_distance_out=float(support_cfg["out_robust_distance"]),
        regime_distance_quantile=float(support_cfg["regime_distance_quantile"]),
    ).fit(train_transformed)
    width_ratio = predictions["interval_width_90"].to_numpy(dtype=float) / max(float(median_width90), 1.0e-9)
    support = support_model.score(transformed_final.reset_index(drop=True), width_ratio)
    result = pd.concat([predictions.reset_index(drop=True), support.reset_index(drop=True)], axis=1)
    result = apply_abstention(result, config["abstention_settings"])
    result = assign_maintenance_recommendations(result, config["maintenance_thresholds"], "lower_90")
    result.attrs["support_metadata"] = support_model.metadata()
    return result


def write_design_note(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Multidomain Deep RUL Design

Phase 5 evaluates whether compact temporal PyTorch models using recent sensor history can improve over the frozen classical AeroGuard benchmarks. All four C-MAPSS training subsets are used for deep model fitting and selection; all four test subsets are held-out benchmark test sets that were observed in earlier phases and are not used for architecture, epoch, preprocessing, conformal, abstention, or maintenance-policy selection.

The feature pipeline reuses Phase 3 regime standardization. Every split fits feature exclusions, operating regimes, normalization, and scaling on fitting engines only. Sequence windows are past-only, left-padded with zeros in standardized space, and include a validity mask channel. Engine-balanced sampling prevents long engines from dominating training.

Six architectures are screened: sequence MLP, causal 1D CNN, unidirectional LSTM, unidirectional GRU, causal residual TCN, and CNN-LSTM. Finalists undergo engine-group cross-validation, then the locked architecture is trained on all four training subsets for the locked epoch count.

Deep uncertainty uses grouped conformal calibration over locked-model out-of-fold validation snapshots, with global and predicted-RUL-band conformal candidates. Support, abstention, and demonstration maintenance recommendations reuse the Phase 4 logic. Recommendations are not approved aircraft-maintenance instructions.
""",
        encoding="utf-8",
        newline="\n",
    )


def write_results_note(path: Path, result: dict[str, Any], config_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Multidomain Deep RUL Results\n\n")
        for key in ["python_version", "torch_version", "cuda_available", "selected_execution_profile", "device", "training_engine_counts", "benchmark_engine_counts", "locked_architecture", "locked_epoch_count", "deep_point_conclusion", "runtime_by_stage"]:
            handle.write(f"- {key}: `{result.get(key)}`\n")
        handle.write(f"- Screening metrics: `{result['screening_metrics']}`\n")
        handle.write(f"- Finalist CV metrics: `{result['finalist_cv_metrics']}`\n")
        handle.write(f"- Benchmark metrics: `{result['benchmark_metrics']}`\n")
        handle.write(f"- Deep uncertainty metrics: `{result['deep_uncertainty_metrics']}`\n")
        handle.write(f"- Maintenance counts: `{result['deep_maintenance_policy_metrics'].get('action_counts')}`\n")
        handle.write("\n## Generated Files\n\n")
        for item in result["generated_files"]:
            handle.write(f"- `{item}`\n")
        handle.write("\n## Warnings\n\n")
        for warning in result["warnings"]:
            handle.write(f"- {warning}\n")
        handle.write("\n## Reproduction Command\n\n```powershell\n")
        handle.write('$env:PYTHONPATH = ".\\src"\n')
        handle.write(
            "python -m aeroguard.pipelines.train_multidomain_deep_rul "
            f'--config "{config_path.as_posix()}"\n'
        )
        handle.write("```\n")


def make_figures(
    output_dir: Path,
    screening: pd.DataFrame,
    cv: pd.DataFrame,
    benchmark: pd.DataFrame,
    point_comparison: pd.DataFrame,
    uncertainty_comparison: pd.DataFrame,
    uncertainty: pd.DataFrame,
    efficiency: pd.DataFrame,
    config: dict[str, Any],
) -> list[str]:
    figures = []
    fig_dir = output_dir / "figures"
    ex_dir = output_dir / "engine_examples"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ex_dir.mkdir(parents=True, exist_ok=True)

    def save(path: Path) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        figures.append(str(path))

    for metric, filename in [("validation_rmse", "screening_rmse_comparison.png"), ("validation_mae", "mae_by_architecture.png"), ("validation_nasa_score", "nasa_score_by_architecture.png")]:
        plt.figure(figsize=(8, 5))
        plt.bar(screening["model_id"], screening[metric])
        plt.xticks(rotation=30, ha="right")
        save(fig_dir / filename)
    plt.figure(figsize=(8, 5)); plt.bar(cv["model_id"], cv["validation_rmse"]); plt.xticks(rotation=30, ha="right"); save(fig_dir / "finalist_cross_validation_comparison.png")
    plt.figure(figsize=(8, 5)); plt.scatter(efficiency["parameter_count"], efficiency["validation_rmse"]); plt.xlabel("Parameters"); plt.ylabel("RMSE"); save(fig_dir / "parameter_count_vs_rmse.png")
    plt.figure(figsize=(8, 5)); plt.scatter(efficiency["cpu_batch_one_median_latency_ms"], efficiency["validation_rmse"]); plt.xlabel("CPU latency ms"); plt.ylabel("RMSE"); save(fig_dir / "cpu_latency_vs_rmse.png")
    plt.figure(figsize=(8, 5)); point_comparison.pivot(index="subset", columns="model", values="mae").plot(kind="bar", ax=plt.gca()); save(fig_dir / "classical_vs_deep_performance_by_subset.png")
    plt.figure(figsize=(7, 6)); plt.scatter(benchmark["true_rul"], benchmark["predicted_rul"], s=12, alpha=0.5); plt.xlabel("True RUL"); plt.ylabel("Predicted RUL"); save(fig_dir / "predicted_vs_true_locked_deep.png")
    plt.figure(figsize=(8, 5)); benchmark["residual"].plot(kind="hist", bins=30); save(fig_dir / "residual_distribution.png")
    for group, filename in [("true_rul_band", "error_by_rul_band.png"), ("operating_regime", "error_by_operating_regime.png"), ("sequence_length_group", "error_by_sequence_length.png")]:
        plt.figure(figsize=(8, 5)); benchmark.groupby(group)["absolute_error"].mean().plot(kind="bar"); plt.xticks(rotation=30, ha="right"); save(fig_dir / filename)
    plt.figure(figsize=(7, 5)); levels=[80,90,95]; plt.plot(levels,[uncertainty[f"covered_{p}"].mean() for p in levels],marker="o"); plt.plot(levels,[p/100 for p in levels],linestyle="--"); save(fig_dir / "deep_uncertainty_coverage_vs_nominal.png")
    plt.figure(figsize=(8, 5)); uncertainty.groupby("subset")["interval_width_90"].mean().plot(kind="bar"); save(fig_dir / "deep_interval_width_by_subset.png")
    plt.figure(figsize=(8, 5)); uncertainty_comparison.pivot(index="subset", columns="model", values="mean_width_90").plot(kind="bar", ax=plt.gca()); save(fig_dir / "classical_vs_deep_interval_width.png")
    plt.figure(figsize=(8, 5)); uncertainty_comparison.pivot(index="subset", columns="model", values="coverage_90").plot(kind="bar", ax=plt.gca()); save(fig_dir / "classical_vs_deep_calibration.png")
    plt.figure(figsize=(8, 5)); uncertainty.groupby("support_status")["abstain_flag"].mean().plot(kind="bar"); save(fig_dir / "abstention_tradeoff.png")
    plt.figure(figsize=(8, 5)); screening.set_index("model_id")["training_seconds"].plot(kind="bar"); plt.xticks(rotation=30, ha="right"); save(fig_dir / "training_runtime_by_architecture.png")
    plt.figure(figsize=(8, 5)); screening.set_index("model_id")["best_epoch"].plot(kind="bar"); plt.xticks(rotation=30, ha="right"); save(fig_dir / "best_epoch_by_architecture.png")
    plt.figure(figsize=(8, 5)); uncertainty["maintenance_action"].value_counts().plot(kind="bar"); plt.xticks(rotation=30, ha="right"); save(fig_dir / "maintenance_action_distribution.png")
    for _, row in pd.concat([benchmark.nsmallest(3, "absolute_error"), benchmark.nlargest(2, "residual"), benchmark.nsmallest(1, "residual"), benchmark[benchmark["subset"] == "FD004"].head(1), benchmark[benchmark["true_rul"] <= 15].head(1)]).drop_duplicates("global_engine_id").head(8).iterrows():
        plt.figure(figsize=(6, 4))
        plt.bar(["true", "pred"], [row["true_rul"], row["predicted_rul"]])
        plt.title(row["global_engine_id"])
        save(ex_dir / f"{row['global_engine_id']}_deep_example.png")
    return figures


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    start = time.perf_counter()
    root = project_root()
    config_path = Path(config_path)
    config = load_config(config_path)
    output_dir = resolve_project_path(config["output_dir"], root)
    checkpoint_dir = resolve_project_path(config["checkpoint_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "engine_examples").mkdir(parents=True, exist_ok=True)
    env = environment_report()
    profile, limits = select_execution_profile(config, env)
    device = device_from_config(config, env)
    config = dict(config)
    config["batch_size"] = limits["effective_batch_size"]
    config["pin_memory"] = bool(device.type == "cuda")
    set_global_seed(int(config["random_seed"]), bool(config.get("deterministic_algorithms", False)))
    print(f"Profile: {profile} ({limits['reason']}); device={device}")
    stage_times = {}
    manifest = create_classical_manifest(output_dir, root, config)

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
    train_dataset, train_meta_windows, train_sequences = make_dataset(screen_train, train_endpoints, pre["features"], spec)
    val_dataset, val_meta_windows, val_sequences = make_dataset(screen_val, val_endpoints, pre["features"], spec)
    sequence_audit(screen_train, train_endpoints, spec, config["rul_bands"]).to_csv(output_dir / "sequence_audit.csv", index=False)
    print("Window creation complete")

    input_dim = train_sequences.shape[2]
    screening_rows = []
    screening_models: dict[str, torch.nn.Module] = {}
    screening_metadata: dict[str, dict[str, Any]] = {}
    for model_config in config["model_registry"]:
        model_id = str(model_config["model_id"])
        model, meta = train_one_model(
            model_config,
            train_dataset,
            val_dataset,
            {**config, "random_seed": int(config["random_seed"]) + len(screening_rows)},
            device,
            int(limits["effective_parameter_budget"]),
            input_dim,
            int(limits["effective_screening_max_epochs"]),
            int(limits["effective_patience"]),
            bool(limits["effective_mixed_precision"]),
        )
        pred_frame = evaluate_model_frame(model, val_dataset, val_meta_windows, device, int(config["batch_size"]), model_id)
        metrics = point_metrics_for_predictions(pred_frame, float(config["abstention_settings"]["high_error_threshold"]))
        row = {
            "model_id": model_id,
            "architecture": model_config["architecture"],
            "parameter_count": trainable_parameter_count(model),
            "best_epoch": meta["best_epoch"],
            "training_seconds": meta["training_seconds"],
            "validation_mae": metrics["mae"],
            "validation_rmse": metrics["rmse"],
            "validation_nasa_score": metrics["nasa_score"],
            "validation_optimistic_rate": metrics["optimistic_prediction_rate"],
        }
        screening_rows.append(row)
        screening_models[model_id] = deepcopy(model).to("cpu")
        screening_metadata[model_id] = meta
        save_checkpoint(checkpoint_dir / f"screening_{model_id}.pt", model.to("cpu"), {"model_config": model_config, **meta})
        model.to(device)
        print(f"Screened {model_id}: RMSE={row['validation_rmse']:.3f}")
    screening_df = pd.DataFrame(screening_rows).sort_values(["validation_rmse", "validation_mae", "validation_nasa_score"])
    screening_df.to_csv(output_dir / "screening_metrics.csv", index=False)

    finalist_ids = screening_df.head(int(limits["effective_finalist_count"]))["model_id"].tolist()
    splits = stratified_engine_group_splits(train, int(limits["effective_finalist_cv_folds"]), 1, [int(config["finalist_cv_seed"])])
    validate_no_engine_leakage(splits)
    write_json(output_dir / "cross_validation_splits.json", {"splits": [split.to_dict() for split in splits], "engine_overlap": []})
    cv_rows, locked_oof_candidates = [], {}
    for model_id in finalist_ids:
        model_config = next(item for item in config["model_registry"] if item["model_id"] == model_id)
        fold_predictions = []
        for fold_index, split in enumerate(splits, start=1):
            fold_pre = fit_preprocessor(train[train["global_engine_id"].isin(split.train_engine_ids)].copy(), config)
            fold_train = apply_preprocessor(fold_pre, train[train["global_engine_id"].isin(split.train_engine_ids)].copy())
            fold_val = apply_preprocessor(fold_pre, train[train["global_engine_id"].isin(split.validation_engine_ids)].copy())
            fold_train_endpoints = build_endpoint_table(fold_train, spec, int(config["maximum_windows_per_engine"]), int(config["finalist_cv_seed"]) + fold_index)
            fold_val_endpoints = snapshot_endpoint_table(fold_val, [float(x) for x in config["validation_snapshot_positions"]])
            fold_train_ds, _, fold_train_seq = make_dataset(fold_train, fold_train_endpoints, fold_pre["features"], spec)
            fold_val_ds, fold_val_meta, _ = make_dataset(fold_val, fold_val_endpoints, fold_pre["features"], spec)
            model, meta = train_one_model(
                model_config,
                fold_train_ds,
                fold_val_ds,
                {**config, "random_seed": int(config["random_seed"]) + 100 + fold_index},
                device,
                int(limits["effective_parameter_budget"]),
                fold_train_seq.shape[2],
                int(limits["effective_finalist_max_epochs"]),
                int(limits["effective_patience"]),
                bool(limits["effective_mixed_precision"]),
            )
            pred_frame = evaluate_model_frame(model, fold_val_ds, fold_val_meta, device, int(config["batch_size"]), model_id)
            pred_frame["fold"] = split.split_id
            fold_predictions.append(pred_frame)
            metrics = point_metrics_for_predictions(pred_frame, float(config["abstention_settings"]["high_error_threshold"]))
            cv_rows.append({"model_id": model_id, "fold": split.split_id, "best_epoch": meta["best_epoch"], "parameter_count": trainable_parameter_count(model), **metrics, "training_seconds": meta["training_seconds"]})
            save_checkpoint(checkpoint_dir / f"cv_{model_id}_{split.split_id}.pt", model.to("cpu"), {"model_config": model_config, "fold": split.to_dict(), **meta})
            print(f"Finalist {model_id} fold {fold_index} complete: RMSE={metrics['rmse']:.3f}")
        locked_oof_candidates[model_id] = pd.concat(fold_predictions, ignore_index=True)
    cv_df = pd.DataFrame(cv_rows)
    cv_summary = cv_df.groupby("model_id").agg(validation_rmse=("rmse", "mean"), validation_mae=("mae", "mean"), validation_nasa_score=("nasa_score", "mean"), best_epoch=("best_epoch", "median"), parameter_count=("parameter_count", "max"), training_seconds=("training_seconds", "sum")).reset_index()
    cv_summary.to_csv(output_dir / "finalist_cross_validation_metrics.csv", index=False)
    ranking = cv_summary.sort_values(["validation_rmse", "validation_mae", "validation_nasa_score"]).copy()
    ranking.to_csv(output_dir / "deep_model_ranking.csv", index=False)
    locked_model_id = str(ranking.iloc[0]["model_id"])
    locked_config = next(item for item in config["model_registry"] if item["model_id"] == locked_model_id)
    locked_epoch_count = max(1, int(round(float(ranking.iloc[0]["best_epoch"]))))
    write_json(output_dir / "locked_deep_model.json", {"model": locked_config, "locked_epoch_count": locked_epoch_count, "selection_source": "training-engine screening and finalist CV only", "benchmark_tests_used_for_selection": False, "ranking_row": ranking.iloc[0].to_dict()})
    print(f"Locked model: {locked_model_id} for {locked_epoch_count} epochs")

    final_start = time.perf_counter()
    final_pre = fit_preprocessor(train.copy(), config)
    final_train = apply_preprocessor(final_pre, train.copy())
    final_endpoints = build_endpoint_table(final_train, spec, int(config["maximum_windows_per_engine"]), int(config["random_seed"]))
    final_ds, final_meta_windows, final_sequences = make_dataset(final_train, final_endpoints, final_pre["features"], spec)
    final_model = build_model(locked_config, final_sequences.shape[2], int(limits["effective_parameter_budget"]))
    try:
        final_model, final_meta = train_fixed_epochs(final_model, final_ds, config, device, locked_epoch_count, bool(limits["effective_mixed_precision"]))
    except RuntimeError as exc:
        if not bool(limits["effective_mixed_precision"]) or "Non-finite gradients" not in str(exc):
            raise
        set_global_seed(int(config["random_seed"]), bool(config.get("deterministic_algorithms", False)))
        final_model = build_model(locked_config, final_sequences.shape[2], int(limits["effective_parameter_budget"]))
        final_model, final_meta = train_fixed_epochs(final_model, final_ds, config, device, locked_epoch_count, False)
        final_meta["mixed_precision_retry_reason"] = "disabled_after_non_finite_gradients"
    save_checkpoint(checkpoint_dir / "locked_deep_model.pt", final_model.to("cpu"), {"model_config": locked_config, "locked_epoch_count": locked_epoch_count, **final_meta})
    final_model.to(device)
    stage_times["final_training_seconds"] = time.perf_counter() - final_start
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

    locked_oof = locked_oof_candidates[locked_model_id].copy()
    locked_uncertainty, unc_cv_metrics, calibrators = choose_uncertainty_method(locked_oof, config)
    unc_cv_metrics.to_csv(output_dir / "deep_uncertainty_cv_metrics.csv", index=False)
    write_json(output_dir / "locked_deep_uncertainty_method.json", locked_uncertainty)
    cal = calibrators["band"] if calibrators["locked_method_id"] == "deep_predicted_band_conformal" else calibrators["global"]
    uncertainty = add_uncertainty(benchmark, cal, calibrators["locked_method_id"], [float(x) for x in config["nominal_coverage_levels"]])
    median_width90 = float(add_uncertainty(locked_oof, cal, calibrators["locked_method_id"], [float(x) for x in config["nominal_coverage_levels"]])["interval_width_90"].median())
    uncertainty = apply_support_abstention_maintenance(uncertainty, transformed_final, final_pre, final_train, config, median_width90)
    uncertainty.to_csv(output_dir / "deep_uncertainty_predictions.csv", index=False)
    deep_unc_metrics = {
        subset: {
            str(level): interval_metrics(group["true_rul"], group["predicted_rul"], group[f"lower_{int(round(level * 100))}"], group[f"upper_{int(round(level * 100))}"], level)
            for level in config["nominal_coverage_levels"]
        }
        for subset, group in uncertainty.groupby("subset")
    }
    deep_unc_metrics["overall"] = {
        str(level): interval_metrics(uncertainty["true_rul"], uncertainty["predicted_rul"], uncertainty[f"lower_{int(round(level * 100))}"], uncertainty[f"upper_{int(round(level * 100))}"], level)
        for level in config["nominal_coverage_levels"]
    }
    write_json(output_dir / "deep_uncertainty_metrics.json", deep_unc_metrics)
    maintenance_recs = uncertainty[["subset", "global_engine_id", UNIT_COLUMN, "true_rul", "predicted_rul", "lower_90", "upper_90", "abstain_flag", "maintenance_action", "action_basis", "conservative_rul_bound", "nominal_interval_level", "prediction_status", "maintenance_disclaimer"]]
    maintenance_recs.to_csv(output_dir / "deep_maintenance_recommendations.csv", index=False)
    maintenance_metrics = maintenance_policy_metrics(uncertainty)
    write_json(output_dir / "deep_maintenance_policy_metrics.json", maintenance_metrics)
    print("Conformal calibration complete")

    classical = json.loads((resolve_project_path(config["phase4_results_path"], root) / "run_summary.json").read_text(encoding="utf-8"))
    comp_rows = []
    unc_comp_rows = []
    for subset, group in benchmark.groupby("subset"):
        deep_m = point_metrics_for_predictions(group, float(config["abstention_settings"]["high_error_threshold"]))
        classical_m = classical["test_metrics_by_subset"][subset]["point"]
        comp_rows.extend([
            {"subset": subset, "model": "classical_random_forest", "training_domains": "FD001-FD003", "input_type": "final-row tabular", "sequence_length": 1, **classical_m, "parameter_count": None, "serialized_size": None, "training_time": None},
            {"subset": subset, "model": "deep_" + locked_model_id, "training_domains": "FD001-FD004", "input_type": "temporal sequence", "sequence_length": int(config["window_length"]), **deep_m, "parameter_count": int(ranking.iloc[0]["parameter_count"]), "serialized_size": None, "training_time": final_meta["training_seconds"]},
        ])
        for model_name, source in [("classical_uncertainty", classical["test_metrics_by_subset"][subset]["intervals"]), ("deep_uncertainty", deep_unc_metrics[subset])]:
            row = {"subset": subset, "model": model_name}
            row.update({"coverage_80": source["0.8"]["coverage"], "coverage_90": source["0.9"]["coverage"], "coverage_95": source["0.95"]["coverage"], "mean_width_90": source["0.9"]["mean_interval_width"], "median_width_90": source["0.9"]["median_interval_width"], "undercoverage_90": source["0.9"]["undercoverage_amount"], "interval_score_90": source["0.9"]["mean_interval_score"]})
            unc_comp_rows.append(row)
    comparison = pd.DataFrame(comp_rows)
    comparison.to_csv(output_dir / "classical_vs_deep.csv", index=False)
    unc_comparison = pd.DataFrame(unc_comp_rows)
    unc_comparison.to_csv(output_dir / "classical_vs_deep_uncertainty.csv", index=False)
    fd004_delta = classical["test_metrics_by_subset"]["FD004"]["point"]["mae"] - benchmark_metrics["FD004"]["mae"]
    crit = config["deep_improvement_classification"]
    if fd004_delta >= float(crit["clear_improvement_min_fd004_mae_reduction"]):
        point_conclusion = "Clear deep-learning improvement"
    elif fd004_delta >= float(crit["moderate_improvement_min_fd004_mae_reduction"]):
        point_conclusion = "Moderate deep-learning improvement"
    elif abs(fd004_delta) <= float(crit["comparable_abs_fd004_mae_delta"]):
        point_conclusion = "Comparable performance"
    elif fd004_delta < 0:
        point_conclusion = "Classical model remains stronger"
    else:
        point_conclusion = "Inconclusive"

    example_single = torch.as_tensor(final_sequences[:1], dtype=torch.float32)
    example_batch = torch.as_tensor(final_sequences[: min(32, len(final_sequences))], dtype=torch.float32)
    efficiency_rows = []
    for model_id, model in screening_models.items():
        row = model_efficiency_row(model_id, model, example_single, example_batch, device, screening_metadata[model_id], int(config.get("latency_repetitions", 60)))
        row["validation_rmse"] = float(screening_df[screening_df["model_id"] == model_id]["validation_rmse"].iloc[0])
        efficiency_rows.append(row)
    efficiency = pd.DataFrame(efficiency_rows)
    efficiency.to_csv(output_dir / "model_efficiency.csv", index=False)
    comparison.loc[comparison["model"].str.startswith("deep_"), "serialized_size"] = int(efficiency[efficiency["model_id"] == locked_model_id]["serialized_size_bytes"].iloc[0])
    comparison.to_csv(output_dir / "classical_vs_deep.csv", index=False)

    onnx_report = {"onnx_available": False, "exported": False}
    try:
        import onnxruntime as ort  # noqa: F401

        class ExportWrapper(torch.nn.Module):
            def __init__(self, model: torch.nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                lengths = x[:, :, -1].sum(dim=1).long().clamp_min(1)
                return self.model(x, lengths)

        onnx_path = checkpoint_dir / "locked_deep_model.onnx"
        wrapper = ExportWrapper(final_model.to("cpu")).eval()
        torch.onnx.export(wrapper, example_single.cpu(), onnx_path, input_names=["sequence"], output_names=["rul"], dynamic_axes={"sequence": {0: "batch"}, "rul": {0: "batch"}}, opset_version=17)
        onnx_report = {"onnx_available": True, "exported": True, "path": str(onnx_path)}
        final_model.to(device)
    except Exception as exc:
        onnx_report = {"onnx_available": False, "exported": False, "reason": f"{type(exc).__name__}: {exc}"}

    figures = make_figures(output_dir, screening_df, cv_summary, benchmark, comparison, unc_comparison, uncertainty, efficiency, config)
    print("Figure generation complete")

    write_json(output_dir / "deep_model_registry.json", {"models": config["model_registry"], "selected_execution_profile": profile})
    write_json(output_dir / "final_fit_metadata.json", {"preprocessor": {"features": final_pre["features"], "retained_raw_features": final_pre["retained"], "excluded_features": final_pre["excluded"], "normalization": final_pre["normalizer"].metadata()}, "locked_model": locked_config, "locked_epoch_count": locked_epoch_count, "support": uncertainty.attrs.get("support_metadata", {}), "onnx": onnx_report})
    design_note = resolve_project_path(config.get("design_note_path", "notes/multidomain_deep_rul_design.md"), root)
    results_note = resolve_project_path(config.get("results_note_path", "notes/multidomain_deep_rul_results.md"), root)
    write_design_note(design_note)
    stage_times["total_runtime_seconds"] = time.perf_counter() - start
    generated_files = [str(path) for path in [
        output_dir / "classical_benchmark_manifest.json", output_dir / "deep_model_registry.json", output_dir / "screening_split.json", output_dir / "cross_validation_splits.json", output_dir / "sequence_audit.csv", output_dir / "screening_metrics.csv", output_dir / "finalist_cross_validation_metrics.csv", output_dir / "deep_model_ranking.csv", output_dir / "locked_deep_model.json", output_dir / "final_fit_metadata.json", output_dir / "benchmark_predictions.csv", output_dir / "benchmark_metrics.json", output_dir / "metrics_by_subset.csv", output_dir / "metrics_by_rul_band.csv", output_dir / "metrics_by_regime.csv", output_dir / "classical_vs_deep.csv", output_dir / "deep_uncertainty_cv_metrics.csv", output_dir / "locked_deep_uncertainty_method.json", output_dir / "deep_uncertainty_predictions.csv", output_dir / "deep_uncertainty_metrics.json", output_dir / "classical_vs_deep_uncertainty.csv", output_dir / "model_efficiency.csv", output_dir / "deep_maintenance_recommendations.csv", output_dir / "deep_maintenance_policy_metrics.json", output_dir / "run_summary.json"]]
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
        "retained_features": final_pre["features"],
        "sequence_window_count": int(len(final_ds)),
        "architecture_registry": config["model_registry"],
        "screening_metrics": screening_df.to_dict(orient="records"),
        "finalist_cv_metrics": cv_summary.to_dict(orient="records"),
        "locked_architecture": locked_model_id,
        "locked_epoch_count": locked_epoch_count,
        "selection_rationale": ranking.iloc[0].to_dict(),
        "benchmark_metrics": benchmark_metrics,
        "classical_vs_deep": comparison.to_dict(orient="records"),
        "deep_point_conclusion": point_conclusion,
        "model_efficiency": efficiency.to_dict(orient="records"),
        "locked_deep_uncertainty_method": locked_uncertainty,
        "deep_uncertainty_metrics": deep_unc_metrics,
        "classical_vs_deep_uncertainty": unc_comparison.to_dict(orient="records"),
        "abstention_metrics": {subset: abstention_metrics(group, 90, float(config["abstention_settings"]["high_error_threshold"])) for subset, group in uncertainty.groupby("subset")},
        "deep_maintenance_policy_metrics": maintenance_metrics,
        "classical_benchmark_manifest": manifest,
        "generated_files": generated_files,
        "warnings": [
            "Benchmark test subsets were already observed in earlier phases and were not used for model selection.",
            "Deep model outputs are research demonstration results, not deployment or aircraft-maintenance instructions.",
            "Training target is capped RUL, while primary benchmark metrics use uncapped final RUL.",
        ],
    }
    write_results_note(results_note, result, config_path)
    write_json(output_dir / "run_summary.json", result)
    print(json.dumps(_json_ready(result), indent=2, allow_nan=False))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train multidomain deep temporal RUL models.")
    parser.add_argument("--config", required=True, help="Path to deep RUL YAML config.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
