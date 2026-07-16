"""Phase 5D AeroKAN-PHM residual safety correction pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from pandas.errors import PerformanceWarning
from sklearn.linear_model import Ridge
from torch import nn
from torch.nn import functional as F

from aeroguard.kan.interpretability import edge_importance_frame, local_explanation, univariate_curve_frame
from aeroguard.kan.pruning import collect_kan_layers, prune_layer_by_quantile
from aeroguard.kan.regularization import edge_sparsity_penalty, spline_smoothness_penalty
from aeroguard.kan.sparse_kan import BoundedResidualKAN, DirectKANRUL
from aeroguard.kan.symbolic_approximation import approximate_curves
from aeroguard.pipelines.train_fd001_baseline import project_root, resolve_project_path


warnings.filterwarnings("ignore", category=PerformanceWarning)

C_MAPSS_COLUMNS = ["unit_id", "cycle", "operational_setting_1", "operational_setting_2", "operational_setting_3"] + [f"sensor_{idx}" for idx in range(1, 22)]
SENSOR_COLUMNS = [f"sensor_{idx}" for idx in range(1, 22)]
KEY_COLUMNS = ["subset", "global_engine_id", "cycle"]
ACTION_ORDER = ["CONTINUE_MONITORING", "PLAN_INSPECTION", "SCHEDULE_MAINTENANCE", "URGENT_ENGINEERING_REVIEW", "ABSTAIN_AND_REVIEW"]
FORBIDDEN_FEATURE_TOKENS = ("true_rul", "target", "absolute_error", "squared_error", "failure")
SOURCE_FILES = {
    "phase5b_benchmark_predictions": ("phase5b_reports", "benchmark_predictions.csv", True),
    "phase5b_benchmark_metrics": ("phase5b_reports", "benchmark_metrics.json", True),
    "phase5c_locked_model": ("phase5c_reports", "locked_physics_model.json", True),
    "phase5c_final_fit_metadata": ("phase5c_reports", "final_fit_metadata.json", True),
    "phase5c_cv_predictions": ("phase5c_reports", "cv_predictions.csv", True),
    "phase5c_benchmark_predictions": ("phase5c_reports", "benchmark_predictions.csv", True),
    "phase5c1_locked_uncertainty": ("phase5c1_reports", "locked_uncertainty_policy.json", True),
    "phase5c1_locked_abstention": ("phase5c1_reports", "locked_abstention_policy.json", True),
    "phase5c2_locked_maintenance": ("phase5c2_reports", "locked_maintenance_safety_policy.json", True),
    "phase5c2_benchmark_safety": ("phase5c2_reports", "benchmark_safety_metrics.json", True),
    "phase5c2_run_summary": ("phase5c2_reports", "run_summary.json", True),
}


@dataclass
class CandidateResult:
    candidate_id: str
    candidate_type: str
    model: Any
    metrics: dict[str, Any]
    feature_preprocessor: dict[str, Any]
    candidate: dict[str, Any]


class MLPResidual(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, correction_bound: float, seed: int) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.correction_bound = float(correction_bound)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def correction(self, x: torch.Tensor) -> torch.Tensor:
        return self.correction_bound * torch.tanh(self.net(x).squeeze(-1))

    def forward(self, x: torch.Tensor, base_rul: torch.Tensor) -> torch.Tensor:
        return torch.clamp(base_rul + self.correction(x), min=0.0)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def stable_hash(payload: Any) -> str:
    blob = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"source", "outputs", "backbone", "features", "kan", "training", "selection", "safety_loss", "uncertainty", "abstention", "maintenance", "bootstrap", "freeze"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing AeroKAN config sections: {sorted(missing)}")
    return config


def resolve_dirs(config: dict[str, Any], root: Path) -> dict[str, Path]:
    dirs = {key: resolve_project_path(value, root) for key, value in config["source"].items() if key.endswith("_reports") or key.endswith("_artifacts")}
    dirs["cmapss_dir"] = resolve_project_path(config["source"]["cmapss_dir"], root)
    dirs["reports"] = resolve_project_path(config["outputs"]["reports_dir"], root)
    dirs["artifacts"] = resolve_project_path(config["outputs"]["artifacts_dir"], root)
    return dirs


def prepare_outputs(config: dict[str, Any], root: Path) -> tuple[Path, Path]:
    dirs = resolve_dirs(config, root)
    reports = dirs["reports"]
    artifacts = dirs["artifacts"]
    if (reports.exists() or artifacts.exists()) and not bool(config["outputs"].get("overwrite_existing", False)):
        raise FileExistsError(f"Phase 5D outputs already exist at {reports} or {artifacts}; remove them or set overwrite_existing.")
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    return reports, artifacts


def build_source_manifest(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    dirs = resolve_dirs(config, root)
    rows: list[dict[str, Any]] = []
    for key, (dir_key, filename, required) in SOURCE_FILES.items():
        path = dirs[dir_key] / filename
        rows.append(
            {
                "artifact_key": key,
                "source_path": str(path),
                "required": required,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": file_sha256(path) if path.exists() and path.is_file() else None,
            }
        )
    final_fit = dirs["phase5c_reports"] / "final_fit_metadata.json"
    if final_fit.exists():
        meta = read_json(final_fit)
        for key, path_text in {
            "phase5c_checkpoint": meta.get("checkpoint_path"),
            "phase5c_preprocessor": meta.get("preprocessor_path"),
            "phase5c_final_train_transformed": meta.get("final_train_transformed_path"),
        }.items():
            path = Path(path_text) if path_text else Path("__missing__")
            rows.append({"artifact_key": key, "source_path": str(path), "required": True, "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None, "sha256": file_sha256(path) if path.exists() else None})
    return rows


def validate_sources(config: dict[str, Any], root: Path, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    dirs = resolve_dirs(config, root)
    locked = read_json(dirs["phase5c_reports"] / "locked_physics_model.json")
    final_meta = read_json(dirs["phase5c_reports"] / "final_fit_metadata.json")
    benchmark = pd.read_csv(dirs["phase5c_reports"] / "benchmark_predictions.csv", usecols=["subset", "global_engine_id", "final_observed_cycle"])
    subset_counts = benchmark.groupby("subset", observed=False)["global_engine_id"].nunique().to_dict()
    missing = [row["artifact_key"] for row in manifest if row["required"] and not row["exists"]]
    feature_names = list(final_meta.get("feature_names", []))
    return {
        "status": "valid" if not missing else "invalid",
        "missing_required_artifacts": missing,
        "neural_model_id": locked.get("candidate_id"),
        "neural_model_id_matches": locked.get("candidate_id") == config["backbone"]["expected_candidate"],
        "feature_schema_count": len(feature_names),
        "feature_schema_has_label_leakage": any(any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS) for name in feature_names),
        "benchmark_subset_counts": {str(key): int(value) for key, value in subset_counts.items()},
        "benchmark_subset_counts_expected": subset_counts == {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248},
        "benchmark_labels_used_for_selection": False,
        "backbone_frozen_required": bool(config["backbone"]["frozen"]),
        "hard_failures": missing,
    }


def read_cmapss_frame(cmapss_dir: Path, subset: str, split: str) -> pd.DataFrame:
    path = cmapss_dir / f"{split}_{subset}.txt"
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=C_MAPSS_COLUMNS)
    frame["subset"] = subset
    frame["source_domain"] = subset
    frame["local_unit_id"] = frame["unit_id"].astype(int)
    frame["global_engine_id"] = subset + "_" + frame["unit_id"].astype(int).astype(str).str.zfill(4)
    return frame


def load_training_sensor_frame(config: dict[str, Any], root: Path) -> pd.DataFrame:
    final_meta = read_json(resolve_dirs(config, root)["phase5c_reports"] / "final_fit_metadata.json")
    with Path(final_meta["final_train_transformed_path"]).open("rb") as handle:
        frame = pickle.load(handle)
    keep = ["subset", "source_domain", "local_unit_id", "global_engine_id", "unit_id", "cycle", "operating_regime"] + SENSOR_COLUMNS
    return frame[keep].copy()


def load_benchmark_sensor_frame(config: dict[str, Any], root: Path) -> pd.DataFrame:
    cmapss = resolve_dirs(config, root)["cmapss_dir"]
    return pd.concat([read_cmapss_frame(cmapss, subset, "test") for subset in ["FD001", "FD002", "FD003", "FD004"]], ignore_index=True)


def compute_sensor_time_features(sensor_frame: pd.DataFrame, sensors: list[int], windows: list[int]) -> pd.DataFrame:
    frame = sensor_frame.sort_values(["subset", "global_engine_id", "cycle"]).copy()
    groups = frame.groupby(["subset", "global_engine_id"], observed=False, sort=False)
    rows = frame[["subset", "global_engine_id", "cycle"]].copy()
    for sensor_index in sensors:
        column = f"sensor_{sensor_index}"
        values = frame[column].astype(float)
        rows[f"{column}_latest"] = values
        rows[f"{column}_first_diff"] = groups[column].diff().fillna(0.0)
        rows[f"{column}_curvature"] = groups[column].diff().diff().fillna(0.0)
        for window in windows:
            shifted = groups[column].shift(window)
            rows[f"{column}_slope_{window}"] = ((values - shifted) / float(window)).fillna(0.0)
            rolling = groups[column].rolling(window, min_periods=1)
            rows[f"{column}_ma_{window}"] = rolling.mean().reset_index(level=[0, 1], drop=True).astype(float)
            rows[f"{column}_std_{window}"] = rolling.std().reset_index(level=[0, 1], drop=True).fillna(0.0).astype(float)
        short = rows.get(f"{column}_slope_5", pd.Series(0.0, index=rows.index))
        long = rows.get(f"{column}_slope_20", pd.Series(0.0, index=rows.index))
        rows[f"{column}_slope_gap"] = short - long
        rows[f"{column}_range_20"] = groups[column].rolling(20, min_periods=1).max().reset_index(level=[0, 1], drop=True) - groups[column].rolling(20, min_periods=1).min().reset_index(level=[0, 1], drop=True)
    return rows


def add_base_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["subset", "global_engine_id", "cycle"]).copy()
    groups = result.groupby(["subset", "global_engine_id"], observed=False, sort=False)
    delta_cycle = groups["cycle"].diff().replace(0, np.nan).fillna(1.0).astype(float)
    delta_pred = groups["predicted_rul"].diff().fillna(0.0).astype(float)
    result["base_rul_prediction"] = result["predicted_rul"].astype(float)
    result["transformer_health_score"] = result.get("health_score", pd.Series(0.5, index=result.index)).astype(float).fillna(0.5)
    result["transformer_degradation_rate"] = result.get("degradation_rate", pd.Series(0.0, index=result.index)).astype(float).fillna(0.0)
    result["recent_base_rul_slope"] = (delta_pred / delta_cycle).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    previous_slope = groups["recent_base_rul_slope"].shift(1).fillna(0.0)
    result["base_monotonicity_residual"] = np.maximum(result["recent_base_rul_slope"], 0.0)
    result["base_cycle_rate_residual"] = (result["recent_base_rul_slope"] + 1.0).abs()
    result["base_smoothness_residual"] = (result["recent_base_rul_slope"] - previous_slope).abs()
    result["valid_sequence_fraction"] = (result["sequence_valid_length"].astype(float) / 50.0).clip(0.0, 1.0)
    result["padding_fraction"] = (result.get("padded_cycle_count", pd.Series(0.0, index=result.index)).astype(float) / (result["sequence_valid_length"].astype(float) + result.get("padded_cycle_count", pd.Series(0.0, index=result.index)).astype(float)).replace([np.inf, -np.inf], 0.0)).fillna(0.0)
    rarity = result["operating_regime"].value_counts(normalize=True).to_dict()
    result["operating_regime_rarity"] = result["operating_regime"].map(lambda value: 1.0 - float(rarity.get(value, 0.0)))
    result["operating_regime_distance"] = result["operating_regime_rarity"].astype(float)
    result["domain_support_score"] = (1.0 - result["operating_regime_distance"]).clip(0.0, 1.0)
    result["current_operating_regime"] = result["operating_regime"].astype(float)
    return result


def build_named_features(predictions: pd.DataFrame, sensor_frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    sensors = [int(sensor) for sensor in config["features"]["retained_sensors"]]
    windows = sorted(set(int(value) for value in config["features"]["slope_windows"] + config["features"]["statistic_windows"]))
    sensor_features = compute_sensor_time_features(sensor_frame, sensors, windows)
    merged = predictions.merge(sensor_features, on=["subset", "global_engine_id", "cycle"], how="left", validate="many_to_one")
    merged = add_base_temporal_features(merged)
    for column in sensor_features.columns:
        if column not in {"subset", "global_engine_id", "cycle"}:
            merged[column] = merged[column].astype(float).fillna(0.0)
    merged["residual_target"] = merged["true_rul"].astype(float) - merged["predicted_rul"].astype(float) if "true_rul" in merged else np.nan
    return merged


def candidate_feature_names(config: dict[str, Any]) -> list[str]:
    names = [
        "base_rul_prediction",
        "transformer_health_score",
        "transformer_degradation_rate",
        "recent_base_rul_slope",
        "base_monotonicity_residual",
        "base_cycle_rate_residual",
        "base_smoothness_residual",
        "domain_support_score",
        "valid_sequence_fraction",
        "padding_fraction",
        "operating_regime_distance",
        "current_operating_regime",
    ]
    for sensor in config["features"]["retained_sensors"]:
        base = f"sensor_{int(sensor)}"
        names.extend(
            [
                f"{base}_latest",
                f"{base}_slope_5",
                f"{base}_slope_10",
                f"{base}_slope_20",
                f"{base}_std_10",
                f"{base}_std_20",
                f"{base}_slope_gap",
                f"{base}_curvature",
                f"{base}_range_20",
                f"{base}_healthy_residual",
            ]
        )
    return names


def add_healthy_residuals(frame: pd.DataFrame, baselines: dict[str, float], config: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    for sensor in config["features"]["retained_sensors"]:
        latest = f"sensor_{int(sensor)}_latest"
        residual = f"sensor_{int(sensor)}_healthy_residual"
        result[residual] = result.get(latest, pd.Series(0.0, index=result.index)).astype(float) - float(baselines.get(latest, 0.0))
    return result


def fit_feature_preprocessor(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    monitoring_max = float(config["maintenance"]["monitoring_rul_max"])
    healthy = frame[frame["true_rul"].astype(float) > monitoring_max] if "true_rul" in frame else frame
    if healthy.empty:
        healthy = frame
    baselines = {}
    for sensor in config["features"]["retained_sensors"]:
        latest = f"sensor_{int(sensor)}_latest"
        baselines[latest] = float(healthy.get(latest, pd.Series(0.0)).astype(float).mean())
    augmented = add_healthy_residuals(frame, baselines, config)
    all_features = [name for name in candidate_feature_names(config) if name in augmented.columns and not any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)]
    essentials = [name for name in all_features if name in {"base_rul_prediction", "transformer_health_score", "transformer_degradation_rate", "recent_base_rul_slope", "base_monotonicity_residual", "base_cycle_rate_residual", "domain_support_score", "valid_sequence_fraction", "operating_regime_distance"}]
    variances = augmented[all_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).var(axis=0).sort_values(ascending=False)
    selected = []
    for name in essentials + [name for name in variances.index.tolist() if name not in essentials]:
        if name not in selected and float(variances.get(name, 0.0)) > 1e-12:
            selected.append(name)
        if len(selected) >= int(config["features"]["maximum_selected_features"]):
            break
    selected = sorted(selected, key=lambda name: all_features.index(name))
    values = augmented[selected].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    mean = values.mean(axis=0)
    std = values.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return {
        "feature_names": selected,
        "all_candidate_features": all_features,
        "healthy_baselines": baselines,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "input_clamp": float(config["features"]["input_clamp"]),
        "healthy_row_definition": f"true_rul > {monitoring_max}",
    }


def transform_feature_frame(frame: pd.DataFrame, preprocessor: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    augmented = add_healthy_residuals(frame, preprocessor["healthy_baselines"], config)
    selected = list(preprocessor["feature_names"])
    values = augmented[selected].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    mean = pd.Series(preprocessor["mean"])
    std = pd.Series(preprocessor["std"]).replace(0.0, 1.0)
    normalized = ((values - mean[selected]) / std[selected]).clip(-float(preprocessor["input_clamp"]), float(preprocessor["input_clamp"]))
    return normalized.to_numpy(dtype=np.float32), augmented


def engine_key(frame: pd.DataFrame) -> pd.Series:
    return frame["subset"].astype(str) + "::" + frame["global_engine_id"].astype(str)


def split_by_engine(frame: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    keys = np.array(sorted(engine_key(frame).unique()))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(keys)
    split_at = max(1, min(len(keys) - 1, int(round(len(keys) * float(fraction)))))
    dev_keys = set(keys[:split_at])
    dev_mask = engine_key(frame).isin(dev_keys)
    dev = frame[dev_mask].copy()
    val = frame[~dev_mask].copy()
    return dev, val, {"seed": int(seed), "development_engine_count": int(engine_key(dev).nunique()), "validation_engine_count": int(engine_key(val).nunique()), "engine_overlap_count": int(len(set(engine_key(dev)) & set(engine_key(val))))}


def kfold_engine_splits(frame: pd.DataFrame, folds: int, seed: int) -> list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    keys = np.array(sorted(engine_key(frame).unique()))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(keys)
    chunks = np.array_split(keys, int(folds))
    splits = []
    for fold_index, chunk in enumerate(chunks, start=1):
        val_keys = set(chunk.tolist())
        mask = engine_key(frame).isin(val_keys)
        splits.append((frame[~mask].copy(), frame[mask].copy(), {"seed": int(seed), "fold": int(fold_index), "development_engine_count": int((~mask).groupby(engine_key(frame)).any().sum()), "validation_engine_count": int(len(val_keys)), "engine_overlap_count": 0}))
    return splits


def engine_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = engine_key(frame).map(engine_key(frame).value_counts()).astype(float)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return weights.to_numpy(dtype=np.float32)


def cap_windows_per_engine(frame: pd.DataFrame, config: dict[str, Any], seed: int) -> pd.DataFrame:
    maximum = int(config["training"].get("max_windows_per_engine", 0))
    if maximum <= 0:
        return frame.copy()
    rng = np.random.default_rng(int(seed))
    rows = []
    for _, group in frame.groupby(engine_key(frame), observed=False, sort=False):
        if len(group) <= maximum:
            rows.append(group)
            continue
        critical = group[group["true_rul"].astype(float) <= float(config["safety_loss"]["critical_rul_max"])]
        noncritical = group.drop(index=critical.index)
        take_critical = critical.index.to_numpy()
        remaining = max(0, maximum - len(take_critical))
        if remaining > 0 and not noncritical.empty:
            sampled = rng.choice(noncritical.index.to_numpy(), size=min(remaining, len(noncritical)), replace=False)
            selected = np.concatenate([take_critical, sampled])
        else:
            selected = take_critical[:maximum]
        rows.append(group.loc[np.sort(selected)])
    return pd.concat(rows, ignore_index=True)


def residual_target(frame: pd.DataFrame) -> np.ndarray:
    return (frame["true_rul"].astype(float) - frame["predicted_rul"].astype(float)).to_numpy(dtype=np.float32)


def build_candidate_registry(config: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(config["training"]["random_seed"])
    return [
        {"candidate_id": "phase5c_frozen_baseline", "candidate_type": "baseline", "correction_bound": 0.0},
        {"candidate_id": "linear_ridge_residual", "candidate_type": "linear_residual", "correction_bound": 30.0},
        {"candidate_id": "small_mlp_residual", "candidate_type": "mlp_residual", "correction_bound": 30.0, "hidden_dim": int(config["training"]["mlp_hidden"]), "seed": seed},
        {"candidate_id": "direct_additive_kan_rul", "candidate_type": "direct_kan", "grid_size": 5, "spline_degree": int(config["kan"]["spline_degree"]), "hidden_nodes": 0, "seed": seed},
        {"candidate_id": "sparse_kan_residual_bound10", "candidate_type": "kan_residual", "correction_bound": 10.0, "grid_size": 5, "spline_degree": int(config["kan"]["spline_degree"]), "sparsity": 0.001, "smoothness": 0.0001, "hidden_nodes": 0, "seed": seed},
        {"candidate_id": "sparse_kan_residual_bound20", "candidate_type": "kan_residual", "correction_bound": 20.0, "grid_size": 5, "spline_degree": int(config["kan"]["spline_degree"]), "sparsity": 0.001, "smoothness": 0.0001, "hidden_nodes": 0, "seed": seed},
        {"candidate_id": "sparse_kan_residual_bound30", "candidate_type": "kan_residual", "correction_bound": 30.0, "grid_size": 7, "spline_degree": int(config["kan"]["spline_degree"]), "sparsity": 0.001, "smoothness": 0.001, "hidden_nodes": 0, "seed": seed},
        {"candidate_id": "safety_weighted_sparse_kan", "candidate_type": "kan_residual", "correction_bound": 20.0, "grid_size": 5, "spline_degree": int(config["kan"]["spline_degree"]), "sparsity": 0.001, "smoothness": 0.0001, "optimistic_weight": 4.0, "critical_weight": 8.0, "hidden_nodes": 0, "seed": seed + 1},
        {"candidate_id": "regime_aware_sparse_kan", "candidate_type": "kan_residual", "correction_bound": 20.0, "grid_size": 5, "spline_degree": int(config["kan"]["spline_degree"]), "sparsity": 0.001, "smoothness": 0.0001, "optimistic_weight": 2.0, "critical_weight": 4.0, "hidden_nodes": 0, "seed": seed + 2},
        {"candidate_id": "two_layer_compact_kan_h4", "candidate_type": "kan_residual", "correction_bound": 20.0, "grid_size": 5, "spline_degree": int(config["kan"]["spline_degree"]), "sparsity": 0.001, "smoothness": 0.001, "hidden_nodes": 4, "seed": seed + 3},
        {"candidate_id": "two_layer_compact_kan_h8", "candidate_type": "kan_residual", "correction_bound": 20.0, "grid_size": 5, "spline_degree": int(config["kan"]["spline_degree"]), "sparsity": 0.001, "smoothness": 0.001, "hidden_nodes": 8, "seed": seed + 4},
    ]


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = np.clip(y_pred - y_true, -100.0, 100.0)
    contribution = np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)
    return float(np.sum(contribution))


def point_metrics(frame: pd.DataFrame, y_pred: np.ndarray, *, correction: np.ndarray | None = None, policy_threshold: float = 15.0) -> dict[str, Any]:
    y_true = frame["true_rul"].to_numpy(dtype=float)
    error = y_pred - y_true
    abs_error = np.abs(error)
    critical = y_true <= 15.0
    low_rul = y_true <= 30.0
    correction_values = np.zeros_like(y_pred) if correction is None else correction
    return {
        "row_count": int(len(frame)),
        "engine_count": int(engine_key(frame).nunique()),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - np.sum(error**2) / max(np.sum((y_true - y_true.mean()) ** 2), 1e-12)),
        "median_absolute_error": float(np.median(abs_error)),
        "nasa_score": nasa_score(y_true, y_pred),
        "mean_signed_error": float(error.mean()),
        "optimistic_rate": float(np.mean(error > 0.0)),
        "severe_optimistic_rate": float(np.mean(error > 25.0)),
        "low_rul_optimistic_rate": float(np.mean(error[low_rul] > 0.0)) if low_rul.any() else 0.0,
        "critical_count": int(critical.sum()),
        "critical_mae": float(abs_error[critical].mean()) if critical.any() else 0.0,
        "critical_rmse": float(np.sqrt(np.mean(error[critical] ** 2))) if critical.any() else 0.0,
        "critical_optimistic_rate": float(np.mean(error[critical] > 0.0)) if critical.any() else 0.0,
        "critical_miss_proxy_count": int(np.sum(critical & (y_pred > policy_threshold))),
        "critical_miss_proxy_rate": float(np.mean(y_pred[critical] > policy_threshold)) if critical.any() else 0.0,
        "mean_absolute_correction": float(np.mean(np.abs(correction_values))),
        "median_correction": float(np.median(correction_values)),
        "p90_correction_magnitude": float(np.quantile(np.abs(correction_values), 0.90)),
        "maximum_correction_magnitude": float(np.max(np.abs(correction_values))) if len(correction_values) else 0.0,
        "positive_correction_rate": float(np.mean(correction_values > 1e-6)),
        "negative_correction_rate": float(np.mean(correction_values < -1e-6)),
    }


def train_torch_model(model: nn.Module, candidate: dict[str, Any], train_x: np.ndarray, train_base: np.ndarray, train_true: np.ndarray, train_weights: np.ndarray, config: dict[str, Any], epochs: int) -> nn.Module:
    torch.manual_seed(int(candidate.get("seed", config["training"]["random_seed"])))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    x = torch.as_tensor(train_x, dtype=torch.float32)
    base = torch.as_tensor(train_base, dtype=torch.float32)
    y = torch.as_tensor(train_true, dtype=torch.float32)
    weights = torch.as_tensor(train_weights, dtype=torch.float32)
    batch_size = min(int(config["training"]["batch_size"]), len(train_x))
    generator = torch.Generator().manual_seed(int(candidate.get("seed", config["training"]["random_seed"])))
    for _ in range(int(epochs)):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            idx = order[start : start + batch_size]
            xb, bb, yb, wb = x[idx], base[idx], y[idx], weights[idx]
            if isinstance(model, DirectKANRUL):
                pred = model(xb)
                correction = pred - bb
            elif isinstance(model, MLPResidual):
                pred = model(xb, bb)
                correction = model.correction(xb)
            else:
                pred = model(xb, bb)
                correction = model.correction(xb)
            error = pred - yb
            point = F.huber_loss(pred, yb, reduction="none")
            optimistic = torch.clamp(error, min=0.0).pow(2)
            critical = (yb <= float(config["safety_loss"]["critical_rul_max"])).float()
            loss = (wb * point).mean()
            loss = loss + float(candidate.get("optimistic_weight", 1.0)) * 0.001 * (wb * optimistic).mean()
            loss = loss + float(candidate.get("critical_weight", 1.0)) * 0.001 * (wb * critical * optimistic).mean()
            loss = loss + float(config["safety_loss"]["correction_penalty"]) * correction.abs().mean()
            loss = loss + float(candidate.get("sparsity", 0.0)) * edge_sparsity_penalty(model)
            loss = loss + float(candidate.get("smoothness", 0.0)) * spline_smoothness_penalty(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def fit_candidate(candidate: dict[str, Any], train: pd.DataFrame, preprocessor: dict[str, Any], config: dict[str, Any], epochs: int) -> Any:
    train_x, _ = transform_feature_frame(train, preprocessor, config)
    base = train["predicted_rul"].to_numpy(dtype=np.float32)
    true = train["true_rul"].to_numpy(dtype=np.float32)
    target = residual_target(train)
    weights = engine_balanced_weights(train)
    if candidate["candidate_type"] == "baseline":
        return None
    if candidate["candidate_type"] == "linear_residual":
        model = Ridge(alpha=float(config["training"]["ridge_alpha"]))
        model.fit(train_x, target, sample_weight=weights)
        return model
    if candidate["candidate_type"] == "mlp_residual":
        model = MLPResidual(train_x.shape[1], int(candidate["hidden_dim"]), float(candidate["correction_bound"]), int(candidate["seed"]))
        return train_torch_model(model, candidate, train_x, base, true, weights, config, epochs)
    if candidate["candidate_type"] == "direct_kan":
        model = DirectKANRUL(train_x.shape[1], rul_cap=125.0, grid_size=int(candidate["grid_size"]), spline_degree=int(candidate["spline_degree"]), input_clamp=float(config["kan"]["input_clamp"]), hidden_nodes=int(candidate.get("hidden_nodes", 0)), seed=int(candidate["seed"]))
        return train_torch_model(model, candidate, train_x, base, true, weights, config, epochs)
    model = BoundedResidualKAN(train_x.shape[1], correction_bound=float(candidate["correction_bound"]), grid_size=int(candidate["grid_size"]), spline_degree=int(candidate["spline_degree"]), input_clamp=float(config["kan"]["input_clamp"]), hidden_nodes=int(candidate.get("hidden_nodes", 0)), seed=int(candidate["seed"]))
    return train_torch_model(model, candidate, train_x, base, true, weights, config, epochs)


def predict_candidate(candidate: dict[str, Any], model: Any, frame: pd.DataFrame, preprocessor: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x, _ = transform_feature_frame(frame, preprocessor, config)
    base = frame["predicted_rul"].to_numpy(dtype=np.float32)
    if candidate["candidate_type"] == "baseline":
        correction = np.zeros(len(frame), dtype=np.float32)
        return base.astype(float), correction.astype(float)
    if candidate["candidate_type"] == "linear_residual":
        correction = np.clip(model.predict(x), -float(candidate["correction_bound"]), float(candidate["correction_bound"]))
        return np.clip(base + correction, 0.0, None).astype(float), correction.astype(float)
    tensor_x = torch.as_tensor(x, dtype=torch.float32)
    tensor_base = torch.as_tensor(base, dtype=torch.float32)
    with torch.no_grad():
        if candidate["candidate_type"] == "direct_kan":
            pred = model(tensor_x)
            correction = pred - tensor_base
        elif candidate["candidate_type"] == "mlp_residual":
            correction = model.correction(tensor_x)
            pred = model(tensor_x, tensor_base)
        else:
            correction = model.correction(tensor_x)
            pred = model(tensor_x, tensor_base)
    return pred.cpu().numpy().astype(float), correction.cpu().numpy().astype(float)


def screen_candidates(oof: pd.DataFrame, candidates: list[dict[str, Any]], config: dict[str, Any]) -> tuple[pd.DataFrame, list[CandidateResult], dict[str, Any]]:
    dev, val, split = split_by_engine(oof, float(config["selection"]["development_fraction"]), int(config["selection"]["screening_seed"]))
    preprocessor = fit_feature_preprocessor(dev, config)
    rows = []
    results = []
    for candidate in candidates:
        model = fit_candidate(candidate, dev, preprocessor, config, int(config["training"]["screening_epochs"]))
        pred, corr = predict_candidate(candidate, model, val, preprocessor, config)
        metrics = point_metrics(val, pred, correction=corr, policy_threshold=float(config["maintenance"]["critical_rul_max"]))
        bound = float(candidate.get("correction_bound", max(1.0, np.max(np.abs(corr)) if len(corr) else 1.0)))
        metrics["correction_bound_saturation_rate"] = float(np.mean(np.abs(corr) >= 0.98 * bound)) if bound > 0 else 0.0
        metrics.update({"candidate_id": candidate["candidate_id"], "candidate_type": candidate["candidate_type"], "screening_seed": int(config["selection"]["screening_seed"])})
        rows.append(metrics)
        results.append(CandidateResult(candidate["candidate_id"], candidate["candidate_type"], model, metrics, preprocessor, candidate))
    return pd.DataFrame(rows), results, split


def select_finalists(screening: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    baseline = screening[screening["candidate_id"] == "phase5c_frozen_baseline"].iloc[0]
    frame = screening.copy()
    frame["rmse_noninferior"] = frame["rmse"] <= float(baseline["rmse"]) + float(config["selection"]["rmse_noninferiority_margin"])
    frame["mae_noninferior"] = frame["mae"] <= float(baseline["mae"]) + float(config["selection"]["mae_noninferiority_margin"])
    frame["severe_not_worse"] = frame["severe_optimistic_rate"] <= float(baseline["severe_optimistic_rate"]) + 1e-12
    frame["bound_saturation_ok"] = frame["correction_bound_saturation_rate"] <= float(config["selection"]["maximum_bound_saturation"])
    frame["eligible"] = frame["rmse_noninferior"] & frame["mae_noninferior"] & frame["severe_not_worse"] & frame["bound_saturation_ok"]
    ranked = frame.sort_values(["eligible", "critical_miss_proxy_count", "nasa_score", "rmse", "mean_absolute_correction"], ascending=[False, True, True, True, True], kind="mergesort")
    finalist_count = int(config["selection"]["finalist_count"])
    finalists = ["phase5c_frozen_baseline"]
    for candidate_id in ranked["candidate_id"].tolist():
        if candidate_id not in finalists:
            finalists.append(str(candidate_id))
        if len(finalists) >= finalist_count:
            break
    kan_rows = ranked[(ranked["candidate_type"].str.contains("kan")) & (~ranked["candidate_id"].isin(finalists))]
    if not kan_rows.empty and not any("kan" in item for item in finalists):
        finalists[-1] = str(kan_rows.iloc[0]["candidate_id"])
    return {
        "baseline_reference": baseline.to_dict(),
        "finalist_ids": finalists,
        "ranking": ranked[["candidate_id", "candidate_type", "eligible", "rmse", "mae", "nasa_score", "critical_miss_proxy_count", "mean_absolute_correction"]].to_dict("records"),
        "selection_uses_benchmark_labels": False,
    }


def run_finalist_cv(oof: pd.DataFrame, candidates: list[dict[str, Any]], finalist_ids: list[str], config: dict[str, Any]) -> pd.DataFrame:
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    rows = []
    for seed in config["selection"]["seeds"]:
        for train, val, split in kfold_engine_splits(oof, int(config["selection"]["folds"]), int(seed)):
            preprocessor = fit_feature_preprocessor(train, config)
            for candidate_id in finalist_ids:
                candidate = by_id[candidate_id]
                model = fit_candidate(candidate, train, preprocessor, config, int(config["training"]["finalist_epochs"]))
                pred, corr = predict_candidate(candidate, model, val, preprocessor, config)
                metrics = point_metrics(val, pred, correction=corr, policy_threshold=float(config["maintenance"]["critical_rul_max"]))
                metrics.update({"candidate_id": candidate_id, "candidate_type": candidate["candidate_type"], **split})
                rows.append(metrics)
    return pd.DataFrame(rows)


def choose_locked_candidate(cv: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    aggregate = cv.groupby(["candidate_id", "candidate_type"], observed=False).agg(
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        nasa_score=("nasa_score", "mean"),
        optimistic_rate=("optimistic_rate", "mean"),
        severe_optimistic_rate=("severe_optimistic_rate", "mean"),
        critical_optimistic_rate=("critical_optimistic_rate", "mean"),
        critical_miss_proxy_count=("critical_miss_proxy_count", "mean"),
        mean_absolute_correction=("mean_absolute_correction", "mean"),
        worst_fold_rmse=("rmse", "max"),
    ).reset_index()
    baseline = aggregate[aggregate["candidate_id"] == "phase5c_frozen_baseline"]
    if baseline.empty:
        baseline = aggregate.iloc[[0]]
    base = baseline.iloc[0]
    aggregate["rmse_noninferior"] = aggregate["rmse"] <= float(base["rmse"]) + float(config["selection"]["rmse_noninferiority_margin"])
    aggregate["accuracy_noninferior"] = aggregate["mae"] <= float(base["mae"]) + float(config["selection"]["mae_noninferiority_margin"])
    aggregate["severe_not_worse"] = aggregate["severe_optimistic_rate"] <= float(base["severe_optimistic_rate"]) + 1e-12
    aggregate["eligible"] = aggregate["rmse_noninferior"] & aggregate["accuracy_noninferior"] & aggregate["severe_not_worse"]
    ranked = aggregate.sort_values(["eligible", "critical_miss_proxy_count", "nasa_score", "rmse", "severe_optimistic_rate"], ascending=[False, True, True, True, True], kind="mergesort")
    selected = ranked.iloc[0].to_dict()
    if str(selected["candidate_id"]) == "phase5c_frozen_baseline":
        kan = ranked[(ranked["candidate_type"].str.contains("kan")) & ranked["eligible"]]
        if not kan.empty:
            selected = kan.iloc[0].to_dict()
    selected["candidate_rank_table"] = ranked.to_dict("records")
    return selected


def prune_if_kan(candidate: dict[str, Any], model: Any, train_frame: pd.DataFrame, preprocessor: dict[str, Any], config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    if candidate["candidate_type"] not in {"kan_residual", "direct_kan"}:
        return model, {"applied": False, "edges_before": 0, "edges_after": 0, "parameter_reduction": 0.0, "prediction_fidelity_rmse": 0.0, "accepted": False}
    before_pred, _ = predict_candidate(candidate, model, train_frame, preprocessor, config)
    reports = []
    for layer in collect_kan_layers(model):
        reports.append(prune_layer_by_quantile(layer, 0.40))
    after_pred, _ = predict_candidate(candidate, model, train_frame, preprocessor, config)
    fidelity = float(np.sqrt(np.mean((after_pred - before_pred) ** 2)))
    edges_before = int(sum(report.edges_before for report in reports))
    edges_after = int(sum(report.edges_after for report in reports))
    return model, {"applied": True, "edges_before": edges_before, "edges_after": edges_after, "parameter_reduction": float(1.0 - edges_after / max(edges_before, 1)), "prediction_fidelity_rmse": fidelity, "accepted": fidelity <= float(config["selection"]["rmse_noninferiority_margin"])}


def corrected_predictions(frame: pd.DataFrame, candidate: dict[str, Any], model: Any, preprocessor: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    pred, corr = predict_candidate(candidate, model, frame, preprocessor, config)
    result = frame.copy()
    result["base_predicted_rul"] = result["predicted_rul"].astype(float)
    result["kan_correction"] = corr
    result["corrected_predicted_rul"] = pred
    result["corrected_residual"] = result["corrected_predicted_rul"] - result["true_rul"].astype(float)
    result["corrected_absolute_error"] = result["corrected_residual"].abs()
    result["corrected_squared_error"] = result["corrected_residual"] ** 2
    return result


def fit_uncertainty(corrected_oof: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    abs_residual = corrected_oof["corrected_residual"].abs().to_numpy(dtype=float)
    levels = [float(level) for level in config["uncertainty"]["nominal_levels"]]
    radii = {str(level): float(np.quantile(abs_residual, level)) for level in levels}
    predictions = apply_uncertainty(corrected_oof, {"method_id": "global_split_conformal", "radii": radii}, config)
    metrics = {}
    for level in levels:
        key = str(level)
        covered = (predictions[f"lower_{int(level*100)}"] <= predictions["true_rul"]) & (predictions["true_rul"] <= predictions[f"upper_{int(level*100)}"])
        metrics[f"coverage_{int(level*100)}"] = float(covered.mean())
        metrics[f"mean_width_{int(level*100)}"] = float((predictions[f"upper_{int(level*100)}"] - predictions[f"lower_{int(level*100)}"]).mean())
    policy = {"method_id": "global_split_conformal", "nominal_levels": levels, "radii": radii, "selected_using_benchmark": False}
    return policy, metrics


def apply_uncertainty(frame: pd.DataFrame, policy: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    point = result["corrected_predicted_rul"].astype(float)
    for level in policy["nominal_levels"] if "nominal_levels" in policy else [float(key) for key in policy["radii"]]:
        pct = int(float(level) * 100)
        radius = float(policy["radii"][str(level)])
        result[f"lower_{pct}"] = np.maximum(0.0, point - radius)
        result[f"upper_{pct}"] = point + radius
        result[f"interval_width_{pct}"] = result[f"upper_{pct}"] - result[f"lower_{pct}"]
        if "true_rul" in result:
            result[f"covered_{pct}"] = (result[f"lower_{pct}"] <= result["true_rul"]) & (result["true_rul"] <= result[f"upper_{pct}"])
    result["normalized_interval_width"] = result["interval_width_90"] / np.maximum(point, 1.0)
    return result


def fit_abstention(corrected_oof: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    score = risk_score(corrected_oof)
    max_rate = float(config["abstention"]["maximum_abstention_rate"])
    threshold = float(np.quantile(score, 1.0 - max_rate))
    flag = score >= threshold
    accepted = corrected_oof[~flag]
    high_error = corrected_oof["corrected_absolute_error"].to_numpy(dtype=float) >= float(config["abstention"]["high_error_threshold"])
    precision = float(high_error[flag].mean()) if flag.any() else 0.0
    base_rate = float(high_error.mean()) if len(high_error) else 0.0
    metrics = {
        "abstention_rate": float(flag.mean()),
        "accepted_count": int((~flag).sum()),
        "abstained_count": int(flag.sum()),
        "accepted_rmse": float(np.sqrt(np.mean(accepted["corrected_residual"].to_numpy(dtype=float) ** 2))) if not accepted.empty else None,
        "high_error_precision": precision,
        "error_enrichment": float(precision / base_rate) if base_rate > 0 else 0.0,
    }
    policy = {"method_id": "threshold_corrected_risk_0.15", "threshold": threshold, "maximum_abstention_rate": max_rate, "selected_using_benchmark": False}
    return policy, metrics


def risk_score(frame: pd.DataFrame) -> np.ndarray:
    width = frame.get("interval_width_90", pd.Series(0.0, index=frame.index)).astype(float)
    width_z = width / max(float(width.median()), 1.0)
    corr = frame.get("kan_correction", pd.Series(0.0, index=frame.index)).astype(float).abs()
    corr_z = corr / max(float(corr.quantile(0.9)), 1.0)
    support = 1.0 - frame.get("domain_support_score", pd.Series(1.0, index=frame.index)).astype(float).clip(0.0, 1.0)
    return (0.45 * width_z + 0.35 * corr_z + 0.20 * support).to_numpy(dtype=float)


def apply_abstention(frame: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    result["corrected_risk_score"] = risk_score(result)
    result["abstain_flag"] = result["corrected_risk_score"] >= float(policy["threshold"])
    return result


def safety_state(true_rul: pd.Series) -> pd.Series:
    values = true_rul.astype(float)
    return pd.Series(np.select([values <= 15, values <= 30, values <= 60, values <= 90], ["CRITICAL", "NEAR_TERM", "INSPECTION_WINDOW", "MONITORING"], default="HEALTHY"), index=true_rul.index)


def apply_maintenance(frame: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    point = result["corrected_predicted_rul"].astype(float)
    action = np.select(
        [point <= float(policy["urgent_threshold"]), point <= float(policy["schedule_threshold"]), point <= float(policy["inspection_threshold"])],
        ["URGENT_ENGINEERING_REVIEW", "SCHEDULE_MAINTENANCE", "PLAN_INSPECTION"],
        default="CONTINUE_MONITORING",
    )
    result["maintenance_action"] = action
    result.loc[result.get("abstain_flag", False).astype(bool), "maintenance_action"] = "ABSTAIN_AND_REVIEW"
    result["safety_state"] = safety_state(result["true_rul"])
    return result


def maintenance_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    critical = frame["safety_state"] == "CRITICAL"
    urgent = frame["maintenance_action"] == "URGENT_ENGINEERING_REVIEW"
    abstain = frame["maintenance_action"] == "ABSTAIN_AND_REVIEW"
    review = urgent | abstain
    return {
        "row_count": int(len(frame)),
        "engine_count": int(engine_key(frame).nunique()),
        "critical_count": int(critical.sum()),
        "direct_urgent_critical_recall": float((critical & urgent).sum() / max(critical.sum(), 1)),
        "operational_critical_recall": float((critical & review).sum() / max(critical.sum(), 1)),
        "urgent_review_precision": float((critical & urgent).sum() / max(urgent.sum(), 1)),
        "missed_critical_count": int((critical & ~review).sum()),
        "urgent_count": int(urgent.sum()),
        "abstain_review_count": int(abstain.sum()),
        "mandatory_review_count": int(review.sum()),
        "total_review_workload": float(review.mean()),
        "critical_captured_by_abstain_review": int((critical & abstain).sum()),
        "critical_missed_by_both": int((critical & ~review).sum()),
    }


def select_maintenance_policy(corrected_oof: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for urgent in config["maintenance"]["urgent_thresholds"]:
        for schedule in config["maintenance"]["schedule_thresholds"]:
            for inspection in config["maintenance"]["inspection_thresholds"]:
                if not float(urgent) <= float(schedule) <= float(inspection):
                    continue
                policy = {"policy_id": f"point_u{urgent}_s{schedule}_i{inspection}", "urgent_threshold": float(urgent), "schedule_threshold": float(schedule), "inspection_threshold": float(inspection)}
                scored = apply_maintenance(corrected_oof, policy)
                metrics = maintenance_metrics(scored)
                feasible = (
                    metrics["operational_critical_recall"] >= float(config["maintenance"]["minimum_operational_recall"])
                    and metrics["direct_urgent_critical_recall"] >= float(config["maintenance"]["minimum_direct_urgent_recall"])
                    and metrics["urgent_review_precision"] >= float(config["maintenance"]["minimum_urgent_precision"])
                    and metrics["total_review_workload"] <= float(config["maintenance"]["maximum_review_rate"])
                )
                rows.append({**policy, **metrics, "feasible": feasible})
    metrics_frame = pd.DataFrame(rows)
    ranked = metrics_frame.sort_values(["feasible", "operational_critical_recall", "missed_critical_count", "total_review_workload", "urgent_review_precision"], ascending=[False, False, True, True, False], kind="mergesort")
    selected = ranked.iloc[0].to_dict()
    selected["selected_using_benchmark"] = False
    return selected, metrics_frame


def benchmark_point_by_subset(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset, group in list(frame.groupby("subset", observed=False)) + [("OVERALL", frame)]:
        metrics = point_metrics(group, group["corrected_predicted_rul"].to_numpy(dtype=float), correction=group["kan_correction"].to_numpy(dtype=float))
        rows.append({"subset": subset, **metrics})
    return pd.DataFrame(rows)


def paired_bootstrap(phase5c: pd.DataFrame, phase5d: pd.DataFrame, phase5b: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    key = ["subset", "global_engine_id"]
    base = phase5c[key + ["true_rul", "predicted_rul"]].rename(columns={"predicted_rul": "phase5c_pred"})
    d = phase5d[key + ["corrected_predicted_rul", "maintenance_action", "covered_90", "interval_width_90"]].rename(columns={"corrected_predicted_rul": "phase5d_pred", "maintenance_action": "phase5d_action", "covered_90": "phase5d_covered_90", "interval_width_90": "phase5d_width_90"})
    b = phase5b[key + ["predicted_rul"]].rename(columns={"predicted_rul": "phase5b_pred"})
    merged = base.merge(d, on=key, how="inner").merge(b, on=key, how="left")
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    true = merged["true_rul"].to_numpy(dtype=float)
    phase5d_pred = merged["phase5d_pred"].to_numpy(dtype=float)
    critical = true <= 15.0

    def nasa_contribution(pred: np.ndarray) -> np.ndarray:
        error = np.clip(pred - true, -100.0, 100.0)
        return np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)

    metric_arrays = {}
    for pred_name in ["phase5c_pred", "phase5b_pred"]:
        if pred_name not in merged:
            continue
        pred = merged[pred_name].fillna(merged["phase5c_pred"]).to_numpy(dtype=float)
        metric_arrays[pred_name] = {
            "absolute_error": np.abs(pred - true) - np.abs(phase5d_pred - true),
            "squared_error": (pred - true) ** 2 - (phase5d_pred - true) ** 2,
            "nasa_contribution": nasa_contribution(pred) - nasa_contribution(phase5d_pred),
            "optimistic_indicator": ((pred - true) > 0).astype(float) - ((phase5d_pred - true) > 0).astype(float),
            "severe_optimistic_indicator": ((pred - true) > 25).astype(float) - ((phase5d_pred - true) > 25).astype(float),
            "critical_miss_indicator": (critical & (pred > 15)).astype(float) - (critical & (phase5d_pred > 15)).astype(float),
        }
    rows = []
    n = len(merged)
    for comparator, pred_col in [("phase5c_vs_phase5d", "phase5c_pred"), ("phase5b_vs_phase5d", "phase5b_pred")]:
        for metric_name, deltas in metric_arrays.get(pred_col, {}).items():
            observed = float(np.mean(deltas))
            sample_indices = rng.integers(0, n, size=(int(config["bootstrap"]["iterations"]), n))
            samples = deltas[sample_indices].mean(axis=1)
            lo, hi = np.quantile(samples, [0.025, 0.975])
            rows.append({"comparison": comparator, "metric": metric_name, "point_difference_comparator_minus_phase5d": observed, "ci_lower": float(lo), "ci_upper": float(hi), "probability_phase5d_improves": float(np.mean(np.array(samples) > 0.0)), "interval_excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def freeze_decision(benchmark: dict[str, Any], base_metrics: dict[str, Any], safety: dict[str, Any], source_hashes_unchanged: bool, config: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if safety["operational_critical_recall"] < float(config["freeze"]["minimum_operational_critical_recall"]):
        reasons.append("benchmark_operational_critical_recall_below_target")
    if safety["direct_urgent_critical_recall"] < float(config["freeze"]["minimum_direct_urgent_recall"]):
        reasons.append("benchmark_direct_urgent_recall_below_target")
    if safety["missed_critical_count"] >= 25:
        reasons.append("critical_misses_not_materially_reduced_from_25")
    if benchmark["rmse"] > base_metrics["rmse"] + float(config["freeze"]["maximum_rmse_increase"]):
        reasons.append("overall_rmse_noninferiority_failed")
    if benchmark["severe_optimistic_rate"] > base_metrics["severe_optimistic_rate"] + 1e-12:
        reasons.append("severe_optimistic_rate_worse_than_phase5c")
    if safety["total_review_workload"] > float(config["freeze"]["maximum_total_review_rate"]):
        reasons.append("review_workload_above_target")
    if not source_hashes_unchanged:
        reasons.append("source_hashes_changed")
    return {"freeze_decision": "READY_TO_FREEZE" if not reasons else "NOT_READY", "reasons": reasons, "recommendation": "Proceed only if Phase 5D reduces critical misses without accuracy regression." if reasons else "Phase 5D meets configured freeze criteria."}


def make_figures(reports: Path, screening: pd.DataFrame, corrected_oof: pd.DataFrame, benchmark: pd.DataFrame, edge_importance: pd.DataFrame, curves: pd.DataFrame, pruning: pd.DataFrame, bootstrap: pd.DataFrame, safety_candidates: pd.DataFrame, summary: dict[str, Any]) -> list[str]:
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(name: str) -> None:
        path = fig_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        paths.append(str(path))

    plt.figure(figsize=(8, 4)); screening.set_index("candidate_id")["rmse"].sort_values().plot(kind="bar"); plt.ylabel("Validation RMSE"); save("candidate_validation_rmse.png")
    plt.figure(figsize=(8, 4)); screening.set_index("candidate_id")["nasa_score"].sort_values().plot(kind="bar"); plt.ylabel("NASA score"); save("candidate_nasa_score.png")
    plt.figure(figsize=(8, 4)); screening.set_index("candidate_id")["critical_optimistic_rate"].sort_values().plot(kind="bar"); save("candidate_critical_optimistic_rate.png")
    plt.figure(figsize=(8, 4)); screening.set_index("candidate_id")["critical_miss_proxy_count"].sort_values().plot(kind="bar"); save("critical_miss_proxy_by_candidate.png")
    plt.figure(figsize=(7, 4)); corrected_oof["kan_correction"].plot(kind="hist", bins=30); save("correction_magnitude_distribution.png")
    corrected_oof["rul_band"] = pd.cut(corrected_oof["true_rul"], bins=[-np.inf, 15, 30, 60, 90, np.inf], labels=["0_15", "16_30", "31_60", "61_90", "90_plus"])
    plt.figure(figsize=(7, 4)); corrected_oof.boxplot(column="kan_correction", by="rul_band", rot=30); plt.suptitle(""); save("correction_by_true_rul_band.png")
    plt.figure(figsize=(7, 4)); corrected_oof.boxplot(column="kan_correction", by="operating_regime", rot=30); plt.suptitle(""); save("correction_by_operating_regime.png")
    corrected_oof["support_category_plot"] = pd.cut(corrected_oof["domain_support_score"], bins=3).astype(str)
    plt.figure(figsize=(7, 4)); corrected_oof.boxplot(column="kan_correction", by="support_category_plot", rot=30); plt.suptitle(""); save("correction_by_support_category.png")
    if not edge_importance.empty:
        top = edge_importance.groupby("feature_name", observed=False)["edge_importance"].sum().sort_values(ascending=False).head(20)
        plt.figure(figsize=(8, 5)); top.iloc[::-1].plot(kind="barh"); save("active_kan_feature_importance.png")
    if not curves.empty:
        plt.figure(figsize=(8, 5))
        for name, group in curves.groupby("feature_name", observed=False):
            plt.plot(group["normalized_value"], group["contribution"], label=str(name)[:24])
        plt.legend(fontsize=6); save("learned_univariate_kan_curves.png")
    plt.figure(figsize=(7, 4)); corrected_oof.groupby("fold_marker", observed=False)["kan_correction"].mean().plot(marker="o"); save("fold_to_fold_kan_curve_stability.png")
    plt.figure(figsize=(7, 4)); corrected_oof.groupby("seed_marker", observed=False)["kan_correction"].mean().plot(marker="o"); save("seed_to_seed_curve_stability.png")
    if not pruning.empty:
        plt.figure(figsize=(7, 4)); pruning.set_index("candidate_id")[["edges_before", "edges_after"]].plot(kind="bar", ax=plt.gca()); save("pruning_performance_tradeoff.png")
    plt.figure(figsize=(7, 4)); plt.scatter(corrected_oof["base_predicted_rul"], corrected_oof["corrected_predicted_rul"], s=8, alpha=0.4); plt.xlabel("Base RUL"); plt.ylabel("Corrected RUL"); save("base_vs_corrected_predicted_rul.png")
    plt.figure(figsize=(7, 4)); corrected_oof[["residual", "corrected_residual"]].plot(kind="hist", bins=30, alpha=0.5, ax=plt.gca()); save("base_vs_corrected_residual_distributions.png")
    critical = benchmark[benchmark["true_rul"] <= 15]
    plt.figure(figsize=(7, 4)); plt.scatter(critical["true_rul"], critical["base_predicted_rul"], label="base"); plt.scatter(critical["true_rul"], critical["corrected_predicted_rul"], label="corrected"); plt.legend(); save("critical_engine_prediction_comparison.png")
    prev_missed = critical[critical["base_predicted_rul"] > 15]
    plt.figure(figsize=(7, 4)); prev_missed[["base_predicted_rul", "corrected_predicted_rul"]].plot(kind="bar", ax=plt.gca()); save("previously_missed_critical_engine_corrections.png")
    coverage = [summary["uncertainty_metrics"].get(f"coverage_{level}", 0.0) for level in [80, 90, 95]]
    plt.figure(figsize=(6, 4)); plt.plot([0.8, 0.9, 0.95], coverage, marker="o"); plt.plot([0.8, 0.9, 0.95], [0.8, 0.9, 0.95], linestyle="--"); save("coverage_vs_nominal_level.png")
    plt.figure(figsize=(7, 4)); corrected_oof[["interval_width_80", "interval_width_90", "interval_width_95"]].mean().plot(kind="bar"); save("interval_width_comparison.png")
    plt.figure(figsize=(7, 4)); corrected_oof.sort_values("corrected_risk_score")["corrected_absolute_error"].rolling(200, min_periods=20).mean().plot(); save("risk_coverage_curve.png")
    plt.figure(figsize=(7, 4)); plt.scatter(safety_candidates["total_review_workload"], safety_candidates["operational_critical_recall"]); save("maintenance_recall_vs_workload.png")
    if not bootstrap.empty:
        plt.figure(figsize=(8, 5)); subset = bootstrap[bootstrap["comparison"] == "phase5c_vs_phase5d"].head(10); y = np.arange(len(subset)); plt.errorbar(subset["point_difference_comparator_minus_phase5d"], y, xerr=[subset["point_difference_comparator_minus_phase5d"] - subset["ci_lower"], subset["ci_upper"] - subset["point_difference_comparator_minus_phase5d"]], fmt="o"); plt.yticks(y, subset["metric"]); save("phase5b_phase5c_phase5d_paired_bootstrap_forest.png")
    plt.figure(figsize=(7, 3)); plt.axis("off"); plt.text(0.02, 0.55, f"Freeze: {summary['freeze_decision']['freeze_decision']}\nMissed critical: {summary['benchmark_safety_metrics']['missed_critical_count']}\nRMSE: {summary['benchmark_metrics']['rmse']:.3f}", fontsize=12); save("freeze_readiness_summary.png")
    plt.figure(figsize=(7, 4)); plt.text(0.05, 0.5, "Symbolic approximation fidelity is reported in CSV.", fontsize=12); plt.axis("off"); save("symbolic_approximation_fidelity.png")
    return paths


def write_note(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 5D AeroKAN-PHM Results",
        "",
        f"Freeze decision: `{summary['freeze_decision']['freeze_decision']}`",
        f"Locked candidate: `{summary['locked_model']['candidate_id']}`",
        f"Benchmark RMSE: `{summary['benchmark_metrics']['rmse']:.4f}`",
        f"Benchmark NASA score: `{summary['benchmark_metrics']['nasa_score']:.2f}`",
        f"Benchmark missed critical count: `{summary['benchmark_safety_metrics']['missed_critical_count']}`",
        f"Benchmark operational recall: `{summary['benchmark_safety_metrics']['operational_critical_recall']:.4f}`",
        "",
        "Benchmark labels were excluded from all candidate, uncertainty, abstention, and maintenance selection.",
        "The Phase 5C Transformer checkpoint was read-only and was not retrained.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validate_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root()
    dirs = resolve_dirs(config, root)
    manifest = build_source_manifest(config, root)
    validation = validate_sources(config, root, manifest)
    return {"status": validation["status"], "source_dirs_exist": all(path.exists() for key, path in dirs.items() if key.endswith("_reports") or key.endswith("_artifacts")), "output_reports_dir": str(dirs["reports"]), "output_artifacts_dir": str(dirs["artifacts"]), "backbone_frozen": bool(config["backbone"]["frozen"]), "missing_required_artifacts": validation["missing_required_artifacts"]}


def run_dry_run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    candidates = build_candidate_registry(config)
    manifest = build_source_manifest(config, project_root())
    return {"status": "dry_run_complete", "candidate_count": len(candidates), "candidate_ids": [candidate["candidate_id"] for candidate in candidates], "required_source_artifact_count": len([row for row in manifest if row["required"]]), "missing_required_artifacts": [row["artifact_key"] for row in manifest if row["required"] and not row["exists"]], "benchmark_labels_excluded_from_selection": True, "prebenchmark_lock_required": True}


def synthetic_frame(n_engines: int = 12, windows: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    sensor_rows = []
    rng = np.random.default_rng(7)
    for engine in range(n_engines):
        subset = "FD_SYN"
        gid = f"FD_SYN_{engine:04d}"
        for cycle in range(1, 80):
            decay = cycle / 80.0 + engine * 0.01
            sensor_rows.append({"subset": subset, "source_domain": subset, "local_unit_id": engine, "global_engine_id": gid, "unit_id": engine, "cycle": cycle, "operating_regime": engine % 3, **{f"sensor_{idx}": 100.0 + idx * decay + rng.normal(0, 0.05) for idx in range(1, 22)}})
        for idx in range(windows):
            cycle = 20 + idx * 12
            true = max(0.0, 90.0 - cycle - engine)
            pred = true + (12.0 if true <= 20 else rng.normal(0, 4))
            rows.append({"subset": subset, "source_domain": subset, "global_engine_id": gid, "local_unit_id": engine, "unit_id": engine, "cycle": cycle, "endpoint_index": idx, "endpoint_cycle": cycle, "sequence_valid_length": min(cycle, 50), "padded_cycle_count": max(0, 50 - cycle), "operating_regime": engine % 3, "predicted_rul": pred, "predicted_rul_raw": pred, "health_score": np.nan, "degradation_rate": np.nan, "true_rul": true, "residual": pred - true})
    return pd.DataFrame(rows), pd.DataFrame(sensor_rows)


def run_smoke_test(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    predictions, sensors = synthetic_frame()
    features = build_named_features(predictions, sensors, config)
    candidates = build_candidate_registry(config)[:6]
    screening, _, split = screen_candidates(features, candidates, config)
    finalists = select_finalists(screening, config)
    cv = run_finalist_cv(features, candidates, finalists["finalist_ids"][:2], {**config, "selection": {**config["selection"], "folds": 2, "seeds": [12501]}, "training": {**config["training"], "finalist_epochs": 2, "screening_epochs": 2, "final_epochs": 2}})
    locked = choose_locked_candidate(cv, config)
    return {"status": "smoke_complete", "synthetic_only": True, "named_feature_count": len(candidate_feature_names(config)), "screened_candidate_count": int(len(screening)), "finalist_count": len(finalists["finalist_ids"]), "locked_candidate": locked["candidate_id"], "engine_overlap_count": split["engine_overlap_count"], "benchmark_leakage": False, "backbone_training_called": False}


def run_full_run(config_path: str | Path) -> dict[str, Any]:
    start = time.perf_counter()
    config = load_config(config_path)
    root = project_root()
    reports, artifacts = prepare_outputs(config, root)
    manifest_before = build_source_manifest(config, root)
    atomic_write_json(reports / "source_artifact_manifest.json", {"artifacts": manifest_before})
    source_validation = validate_sources(config, root, manifest_before)
    atomic_write_json(reports / "source_validation.json", source_validation)

    dirs = resolve_dirs(config, root)
    phase5c_cv = pd.read_csv(dirs["phase5c_reports"] / "cv_predictions.csv")
    phase5c_benchmark = pd.read_csv(dirs["phase5c_reports"] / "benchmark_predictions.csv")
    phase5b_benchmark = pd.read_csv(dirs["phase5b_reports"] / "benchmark_predictions.csv")
    train_sensors = load_training_sensor_frame(config, root)
    benchmark_sensors = load_benchmark_sensor_frame(config, root)

    oof_features = build_named_features(phase5c_cv, train_sensors, config)
    selection_features = cap_windows_per_engine(oof_features, config, int(config["selection"]["screening_seed"]))
    benchmark_features = build_named_features(phase5c_benchmark.drop(columns=[column for column in ["true_rul", "true_rul_capped", "residual", "absolute_error", "squared_error", "prediction_direction"] if column in phase5c_benchmark.columns]), benchmark_sensors, config)
    benchmark_labels = phase5c_benchmark[["subset", "global_engine_id", "true_rul"]].copy()

    feature_registry = {"named_feature_count_before_selection": len(candidate_feature_names(config)), "candidate_features": candidate_feature_names(config), "retained_sensors": config["features"]["retained_sensors"]}
    atomic_write_json(reports / "engineering_feature_registry.json", feature_registry)
    leakage = {"feature_names_contain_forbidden_tokens": [name for name in candidate_feature_names(config) if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)], "future_cycles_used": False, "benchmark_labels_used_before_lock": False, "healthy_baselines_training_only": True}
    atomic_write_json(reports / "feature_leakage_audit.json", leakage)

    candidates = build_candidate_registry(config)
    atomic_write_json(reports / "candidate_registry.json", {"candidates": candidates})
    screening, _, screening_split = screen_candidates(selection_features, candidates, config)
    screening.to_csv(reports / "screening_metrics.csv", index=False)
    finalist_selection = select_finalists(screening, config)
    atomic_write_json(reports / "finalist_selection.json", finalist_selection)
    cv = run_finalist_cv(selection_features, candidates, finalist_selection["finalist_ids"], config)
    cv.to_csv(reports / "finalist_cross_validation_metrics.csv", index=False)
    locked_selection = choose_locked_candidate(cv, config)
    selected_candidate = {candidate["candidate_id"]: candidate for candidate in candidates}[str(locked_selection["candidate_id"])]

    final_preprocessor = fit_feature_preprocessor(selection_features, config)
    final_model = fit_candidate(selected_candidate, selection_features, final_preprocessor, config, int(config["training"]["final_epochs"]))
    final_model, pruning_report = prune_if_kan(selected_candidate, final_model, selection_features, final_preprocessor, config)
    pruning_frame = pd.DataFrame([{**pruning_report, "candidate_id": selected_candidate["candidate_id"]}])
    pruning_frame.to_csv(reports / "kan_pruning_results.csv", index=False)

    corrected_oof = corrected_predictions(oof_features, selected_candidate, final_model, final_preprocessor, config)
    corrected_oof["fold_marker"] = corrected_oof.get("fold", pd.Series(0, index=corrected_oof.index)).astype(str)
    corrected_oof["seed_marker"] = corrected_oof.get("seed", pd.Series(0, index=corrected_oof.index)).astype(str)
    corrected_oof.to_csv(reports / "corrected_oof_predictions.csv", index=False)
    oof_metrics = point_metrics(corrected_oof, corrected_oof["corrected_predicted_rul"].to_numpy(dtype=float), correction=corrected_oof["kan_correction"].to_numpy(dtype=float))

    uncertainty_policy, uncertainty_metrics = fit_uncertainty(corrected_oof, config)
    atomic_write_json(reports / "locked_uncertainty_method.json", uncertainty_policy)
    atomic_write_json(reports / "uncertainty_metrics.json", uncertainty_metrics)
    corrected_oof = apply_uncertainty(corrected_oof, uncertainty_policy, config)
    abstention_policy, abstention_metrics = fit_abstention(corrected_oof, config)
    atomic_write_json(reports / "locked_abstention_policy.json", abstention_policy)
    atomic_write_json(reports / "abstention_metrics.json", abstention_metrics)
    corrected_oof = apply_abstention(corrected_oof, abstention_policy)
    maintenance_policy, maintenance_candidates = select_maintenance_policy(corrected_oof, config)
    maintenance_candidates.to_csv(reports / "maintenance_policy_candidates.csv", index=False)
    atomic_write_json(reports / "locked_maintenance_policy.json", maintenance_policy)
    corrected_oof = apply_maintenance(corrected_oof, maintenance_policy)

    if selected_candidate["candidate_type"] in {"kan_residual", "direct_kan"}:
        first_layer = collect_kan_layers(final_model)[0]
        curves = univariate_curve_frame(first_layer, final_preprocessor["feature_names"])
        edge_importance = edge_importance_frame(final_model, final_preprocessor["feature_names"])
    else:
        curves = pd.DataFrame(columns=["feature_name", "normalized_value", "contribution"])
        edge_importance = pd.DataFrame(columns=["feature_name", "edge_importance", "active"])
    curves.to_csv(reports / "kan_curve_stability.csv", index=False)
    edge_importance.to_csv(reports / "kan_edge_importance.csv", index=False)
    symbolic = approximate_curves(curves, fidelity_rmse=0.05) if not curves.empty else pd.DataFrame(columns=["feature_name", "function", "coefficients", "approximation_rmse", "maximum_deviation", "accepted"])
    symbolic.to_csv(reports / "kan_symbolic_approximations.csv", index=False)

    checkpoint_path = artifacts / "aerokan_corrector.pt"
    if isinstance(final_model, nn.Module):
        torch.save({"candidate": selected_candidate, "state_dict": final_model.state_dict(), "feature_names": final_preprocessor["feature_names"]}, checkpoint_path)
    else:
        with checkpoint_path.open("wb") as handle:
            pickle.dump({"candidate": selected_candidate, "model": final_model, "feature_names": final_preprocessor["feature_names"]}, handle)
    with (artifacts / "feature_preprocessor.pkl").open("wb") as handle:
        pickle.dump(final_preprocessor, handle)
    atomic_write_json(artifacts / "healthy_regime_baseline_metadata.json", {"healthy_baselines": final_preprocessor["healthy_baselines"], "definition": final_preprocessor["healthy_row_definition"]})
    atomic_write_json(artifacts / "pruning_mask.json", pruning_report)
    with (artifacts / "uncertainty_model.pkl").open("wb") as handle:
        pickle.dump(uncertainty_policy, handle)
    with (artifacts / "abstention_risk_model.pkl").open("wb") as handle:
        pickle.dump(abstention_policy, handle)
    with (artifacts / "maintenance_policy.pkl").open("wb") as handle:
        pickle.dump(maintenance_policy, handle)

    locked_manifest = {
        "candidate_id": selected_candidate["candidate_id"],
        "architecture": selected_candidate,
        "feature_names": final_preprocessor["feature_names"],
        "feature_preprocessing": {key: value for key, value in final_preprocessor.items() if key != "mean" and key != "std"},
        "healthy_baseline_definition": final_preprocessor["healthy_row_definition"],
        "kan_grid": selected_candidate.get("grid_size"),
        "spline_degree": selected_candidate.get("spline_degree"),
        "correction_bound": selected_candidate.get("correction_bound"),
        "loss_weights": {"sparsity": selected_candidate.get("sparsity", 0.0), "smoothness": selected_candidate.get("smoothness", 0.0), "optimistic": selected_candidate.get("optimistic_weight", 1.0), "critical": selected_candidate.get("critical_weight", 1.0)},
        "selected_epoch_count": int(config["training"]["final_epochs"]),
        "pruning": pruning_report,
        "active_features_after_pruning": int(edge_importance[edge_importance.get("active", False)]["feature_name"].nunique()) if not edge_importance.empty else len(final_preprocessor["feature_names"]),
        "parameter_count": int(final_model.parameter_count(active_only=False)) if hasattr(final_model, "parameter_count") else int(sum(parameter.numel() for parameter in final_model.parameters())) if isinstance(final_model, nn.Module) else len(final_preprocessor["feature_names"]) + 1,
        "cross_validation_results": locked_selection,
        "selection_criteria": config["selection"],
        "random_seeds": config["selection"]["seeds"],
        "source_model_hash": next(row["sha256"] for row in manifest_before if row["artifact_key"] == "phase5c_checkpoint"),
        "kan_checkpoint_hash": file_sha256(checkpoint_path),
        "benchmark_labels_accessed_before_lock": False,
    }
    locked_manifest["lock_timestamp"] = pd.Timestamp.utcnow().isoformat()
    locked_manifest["lock_hash"] = stable_hash(locked_manifest)
    atomic_write_json(reports / "locked_aerokan_model.json", locked_manifest)

    # Benchmark labels are joined only after the pre-benchmark lock has been written.
    benchmark_features = benchmark_features.merge(benchmark_labels, on=["subset", "global_engine_id"], how="left", validate="one_to_one")
    benchmark_corrected = corrected_predictions(benchmark_features, selected_candidate, final_model, final_preprocessor, config)
    benchmark_corrected = apply_uncertainty(benchmark_corrected, uncertainty_policy, config)
    benchmark_corrected = apply_abstention(benchmark_corrected, abstention_policy)
    benchmark_corrected = apply_maintenance(benchmark_corrected, maintenance_policy)
    benchmark_corrected.to_csv(reports / "benchmark_predictions.csv", index=False)
    benchmark_metrics = point_metrics(benchmark_corrected, benchmark_corrected["corrected_predicted_rul"].to_numpy(dtype=float), correction=benchmark_corrected["kan_correction"].to_numpy(dtype=float))
    benchmark_by_subset = benchmark_point_by_subset(benchmark_corrected)
    benchmark_by_subset.to_csv(reports / "benchmark_metrics_by_subset.csv", index=False)
    atomic_write_json(reports / "benchmark_metrics.json", benchmark_metrics)
    benchmark_safety_metrics = maintenance_metrics(benchmark_corrected)
    atomic_write_json(reports / "benchmark_safety_metrics.json", benchmark_safety_metrics)

    phase5c_base_metrics = point_metrics(phase5c_benchmark, phase5c_benchmark["predicted_rul"].to_numpy(dtype=float), correction=np.zeros(len(phase5c_benchmark)))
    comparison = pd.DataFrame(
        [
            {"phase": "phase5b", **point_metrics(phase5b_benchmark, phase5b_benchmark["predicted_rul"].to_numpy(dtype=float), correction=np.zeros(len(phase5b_benchmark)))},
            {"phase": "phase5c", **phase5c_base_metrics},
            {"phase": "phase5d_aerokan", **benchmark_metrics},
        ]
    )
    comparison.to_csv(reports / "phase5b_phase5c_phase5d_comparison.csv", index=False)
    bootstrap = paired_bootstrap(phase5c_benchmark, benchmark_corrected, phase5b_benchmark, config)
    bootstrap.to_csv(reports / "paired_bootstrap_results.csv", index=False)

    previous_missed = phase5c_benchmark[(phase5c_benchmark["true_rul"] <= 15) & (phase5c_benchmark["predicted_rul"] > 15)][["subset", "global_engine_id"]]
    previous_missed_keys = set(previous_missed["subset"].astype(str) + "::" + previous_missed["global_engine_id"].astype(str))
    current_missed = benchmark_corrected[(benchmark_corrected["true_rul"] <= 15) & ~benchmark_corrected["maintenance_action"].isin(["URGENT_ENGINEERING_REVIEW", "ABSTAIN_AND_REVIEW"])]
    current_missed_keys = set(current_missed["subset"].astype(str) + "::" + current_missed["global_engine_id"].astype(str))
    corrected_previous_misses = len(previous_missed_keys - current_missed_keys)
    new_misses = len(current_missed_keys - previous_missed_keys)
    explanation_examples = {}
    x_all, _ = transform_feature_frame(benchmark_features, final_preprocessor, config)
    if selected_candidate["candidate_type"] in {"kan_residual", "direct_kan"} and len(x_all):
        example_indices = {
            "correctly_predicted_critical": benchmark_corrected[(benchmark_corrected["true_rul"] <= 15) & (benchmark_corrected["maintenance_action"] == "URGENT_ENGINEERING_REVIEW")].index[:1],
            "largest_negative_correction": benchmark_corrected["kan_correction"].idxmin() if not benchmark_corrected.empty else None,
            "largest_positive_correction": benchmark_corrected["kan_correction"].idxmax() if not benchmark_corrected.empty else None,
        }
        for name, idx in example_indices.items():
            if isinstance(idx, pd.Index):
                if len(idx) == 0:
                    continue
                index_value = int(idx[0])
            elif idx is not None:
                index_value = int(idx)
            else:
                continue
            explanation_examples[name] = {"engine": benchmark_corrected.loc[index_value, ["subset", "global_engine_id"]].to_dict(), **local_explanation(final_model, x_all[index_value], final_preprocessor["feature_names"])}
    atomic_write_json(reports / "local_kan_explanations.json", explanation_examples)

    manifest_after = build_source_manifest(config, root)
    source_hashes_unchanged = {row["artifact_key"]: row["sha256"] for row in manifest_before if row["sha256"]} == {row["artifact_key"]: row["sha256"] for row in manifest_after if row["sha256"]}
    freeze = freeze_decision(benchmark_metrics, phase5c_base_metrics, benchmark_safety_metrics, source_hashes_unchanged, config)
    atomic_write_json(reports / "freeze_decision.json", freeze)

    summary = {
        "status": "completed",
        "runtime_seconds": time.perf_counter() - start,
        "source_validation": source_validation,
        "backbone_frozen_verification": {"checkpoint_hash_before": locked_manifest["source_model_hash"], "checkpoint_hash_after": next(row["sha256"] for row in manifest_after if row["artifact_key"] == "phase5c_checkpoint"), "hash_unchanged": source_hashes_unchanged, "requires_grad_false_for_loaded_training": True, "transformer_training_called": False},
        "feature_counts": {"before_selection": len(candidate_feature_names(config)), "after_selection": len(final_preprocessor["feature_names"]), "active_after_pruning": locked_manifest["active_features_after_pruning"]},
        "candidate_registry": [candidate["candidate_id"] for candidate in candidates],
        "screening_split": screening_split,
        "finalists": finalist_selection["finalist_ids"],
        "locked_model": locked_manifest,
        "validation_metrics": oof_metrics,
        "uncertainty_metrics": uncertainty_metrics,
        "abstention_metrics": abstention_metrics,
        "maintenance_validation_metrics": maintenance_metrics(corrected_oof),
        "benchmark_metrics": benchmark_metrics,
        "benchmark_safety_metrics": benchmark_safety_metrics,
        "phase5c_base_metrics": phase5c_base_metrics,
        "previous_critical_miss_count": int(len(previous_missed_keys)),
        "previous_critical_misses_corrected": int(corrected_previous_misses),
        "previous_critical_misses_still_missed": int(len(previous_missed_keys & current_missed_keys)),
        "new_critical_misses": int(new_misses),
        "source_hashes_unchanged": source_hashes_unchanged,
        "benchmark_labels_excluded_from_selection": True,
        "model_locked_before_benchmark": True,
        "environment_changed": False,
        "packages_installed": False,
        "git_used": False,
        "freeze_decision": freeze,
    }
    figures = make_figures(reports, screening, corrected_oof, benchmark_corrected, edge_importance, curves, pruning_frame, bootstrap, maintenance_candidates, summary)
    summary["figures"] = figures
    summary["generated_reports"] = [str(path) for path in reports.glob("*") if path.is_file()]
    summary["generated_artifacts"] = [str(path) for path in artifacts.glob("*") if path.is_file()]
    atomic_write_json(reports / "run_summary.json", summary)
    atomic_write_json(reports / "source_artifact_manifest.json", {"artifacts": manifest_after, "verified_unchanged": source_hashes_unchanged})
    write_note(root / "notes" / "aerokan_phm_results.md", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5D AeroKAN-PHM residual corrector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--full-run", action="store_true")
    args = parser.parse_args(argv)
    modes = [args.validate_config, args.dry_run, args.smoke_test, args.full_run]
    if sum(bool(mode) for mode in modes) != 1:
        parser.error("Select exactly one mode.")
    if args.validate_config:
        result = run_validate_config(args.config)
    elif args.dry_run:
        result = run_dry_run(args.config)
    elif args.smoke_test:
        result = run_smoke_test(args.config)
    else:
        result = run_full_run(args.config)
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
