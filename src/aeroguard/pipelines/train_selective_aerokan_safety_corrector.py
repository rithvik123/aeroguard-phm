"""Phase 5D.1 selective one-sided AeroKAN safety correction pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import tempfile
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.nn import functional as F

from aeroguard.kan.interpretability import edge_importance_frame, local_explanation, univariate_curve_frame
from aeroguard.kan.pruning import collect_kan_layers, prune_layer_by_quantile
from aeroguard.kan.regularization import edge_sparsity_penalty, spline_smoothness_penalty
from aeroguard.kan.sparse_kan import SparseKANRegressor
from aeroguard.kan.symbolic_approximation import approximate_curves
from aeroguard.pipelines.train_aerokan_rul_corrector import (
    FORBIDDEN_FEATURE_TOKENS,
    add_healthy_residuals,
    apply_abstention,
    apply_uncertainty,
    benchmark_point_by_subset,
    build_named_features,
    cap_windows_per_engine,
    engine_balanced_weights,
    engine_key,
    file_sha256,
    fit_abstention,
    fit_uncertainty,
    json_ready,
    kfold_engine_splits,
    load_benchmark_sensor_frame,
    load_training_sensor_frame,
    maintenance_metrics,
    nasa_score,
    point_metrics,
    read_json,
    safety_state,
    select_maintenance_policy,
    split_by_engine,
    stable_hash,
    synthetic_frame as aerokan_synthetic_frame,
)
from aeroguard.pipelines.train_fd001_baseline import project_root, resolve_project_path


warnings.filterwarnings("ignore", category=ConvergenceWarning)

SOURCE_FILES = {
    "phase5c_locked_model": ("phase5c_reports", "locked_physics_model.json", True),
    "phase5c_final_fit_metadata": ("phase5c_reports", "final_fit_metadata.json", True),
    "phase5c_cv_predictions": ("phase5c_reports", "cv_predictions.csv", True),
    "phase5c_benchmark_predictions": ("phase5c_reports", "benchmark_predictions.csv", True),
    "phase5c1_locked_uncertainty": ("phase5c1_reports", "locked_uncertainty_policy.json", True),
    "phase5c1_locked_abstention": ("phase5c1_reports", "locked_abstention_policy.json", True),
    "phase5c2_locked_maintenance": ("phase5c2_reports", "locked_maintenance_safety_policy.json", True),
    "phase5c2_benchmark_safety": ("phase5c2_reports", "benchmark_safety_metrics.json", True),
    "phase5d_locked_model": ("phase5d_reports", "locked_aerokan_model.json", True),
    "phase5d_benchmark_predictions": ("phase5d_reports", "benchmark_predictions.csv", True),
    "phase5d_run_summary": ("phase5d_reports", "run_summary.json", True),
    "phase5d_checkpoint": ("phase5d_artifacts", "aerokan_corrector.pt", True),
    "phase5d_preprocessor": ("phase5d_artifacts", "feature_preprocessor.pkl", True),
}

BENCHMARK_LABEL_OR_ERROR_COLUMNS = {
    "true_rul",
    "true_rul_capped",
    "target_rul_capped",
    "target_rul_uncapped",
    "residual",
    "absolute_error",
    "squared_error",
    "prediction_direction",
}

TRANSFORMER_TRAINING_CALLED = False


@dataclass
class GateFit:
    candidate: dict[str, Any]
    model: Any
    preprocessor: dict[str, Any]
    threshold: float
    metrics: dict[str, Any]


@dataclass
class CorrectionFit:
    candidate: dict[str, Any]
    model: Any
    preprocessor: dict[str, Any]
    metrics: dict[str, Any]


class NoGateModel:
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.ones(len(x), dtype=float)


class ConstantProbabilityGateModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        positive = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


class RuleGateModel:
    def __init__(self, weights: dict[str, float], score_min: float, score_max: float) -> None:
        self.weights = weights
        self.score_min = float(score_min)
        self.score_max = float(score_max)

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        score = np.zeros(len(frame), dtype=float)
        for name, weight in self.weights.items():
            score += float(weight) * frame.get(name, pd.Series(0.0, index=frame.index)).astype(float).to_numpy()
        return score

    def predict_proba_from_frame(self, frame: pd.DataFrame) -> np.ndarray:
        score = self.score_frame(frame)
        denom = max(self.score_max - self.score_min, 1e-6)
        return np.clip((score - self.score_min) / denom, 0.0, 1.0)


class IsotonicLogisticGateModel:
    def __init__(self, logistic: LogisticRegression, calibrator: IsotonicRegression) -> None:
        self.logistic = logistic
        self.calibrator = calibrator

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        base = self.logistic.predict_proba(x)[:, 1]
        return np.clip(self.calibrator.predict(base), 0.0, 1.0)


class SparseKANGateModel(nn.Module):
    def __init__(self, input_dim: int, *, grid_size: int, spline_degree: int, input_clamp: float, seed: int) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.kan = SparseKANRegressor(input_dim, grid_size=grid_size, spline_degree=spline_degree, input_clamp=input_clamp, hidden_nodes=0, seed=seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(x)

    def probability(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def parameter_count(self, *, active_only: bool = False) -> int:
        return self.kan.parameter_count(active_only=active_only)


class ConstantMagnitudeModel:
    def __init__(self, magnitude: float, bound: float) -> None:
        self.magnitude_value = float(np.clip(magnitude, 0.0, bound))
        self.correction_bound = float(bound)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.magnitude_value, dtype=float)


class LinearNonNegativeMagnitudeModel:
    def __init__(self, regressor: LinearRegression, bound: float) -> None:
        self.regressor = regressor
        self.correction_bound = float(bound)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.regressor.predict(x), 0.0, self.correction_bound)


class MLPMagnitudeModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bound: float, seed: int) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.correction_bound = float(bound)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def magnitude(self, x: torch.Tensor) -> torch.Tensor:
        return self.correction_bound * torch.sigmoid(self.net(x).squeeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.magnitude(x)


class OneSidedKANMagnitude(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        correction_bound: float,
        grid_size: int = 5,
        spline_degree: int = 3,
        input_clamp: float = 5.0,
        hidden_nodes: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.correction_bound = float(correction_bound)
        self.kan = SparseKANRegressor(
            input_dim,
            grid_size=int(grid_size),
            spline_degree=int(spline_degree),
            input_clamp=float(input_clamp),
            hidden_nodes=int(hidden_nodes),
            seed=int(seed),
        )

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(x)

    def magnitude(self, x: torch.Tensor) -> torch.Tensor:
        return self.correction_bound * torch.sigmoid(self.raw(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.magnitude(x)

    def parameter_count(self, *, active_only: bool = False) -> int:
        return self.kan.parameter_count(active_only=active_only)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "source",
        "outputs",
        "backbone",
        "features",
        "dangerous_optimism",
        "gate",
        "correction",
        "training",
        "selection",
        "safety_loss",
        "uncertainty",
        "abstention",
        "maintenance",
        "bootstrap",
        "freeze",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing selective AeroKAN config sections: {sorted(missing)}")
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
        raise FileExistsError(f"Selective Phase 5D.1 outputs already exist at {reports} or {artifacts}; remove them or set overwrite_existing.")
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
                "required": bool(required),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": file_sha256(path) if path.exists() and path.is_file() else None,
            }
        )
    final_fit = dirs["phase5c_reports"] / "final_fit_metadata.json"
    if final_fit.exists():
        meta = read_json(final_fit)
        for key, value in {
            "phase5c_checkpoint": meta.get("checkpoint_path"),
            "phase5c_preprocessor": meta.get("preprocessor_path"),
            "phase5c_final_train_transformed": meta.get("final_train_transformed_path"),
        }.items():
            path = Path(value) if value else Path("__missing__")
            rows.append(
                {
                    "artifact_key": key,
                    "source_path": str(path),
                    "required": True,
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else None,
                    "sha256": file_sha256(path) if path.exists() and path.is_file() else None,
                }
            )
    return rows


def validate_sources(config: dict[str, Any], root: Path, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    dirs = resolve_dirs(config, root)
    missing = [row["artifact_key"] for row in manifest if row["required"] and not row["exists"]]
    locked = read_json(dirs["phase5c_reports"] / "locked_physics_model.json") if (dirs["phase5c_reports"] / "locked_physics_model.json").exists() else {}
    phase5d_locked = read_json(dirs["phase5d_reports"] / "locked_aerokan_model.json") if (dirs["phase5d_reports"] / "locked_aerokan_model.json").exists() else {}
    final_meta = read_json(dirs["phase5c_reports"] / "final_fit_metadata.json") if (dirs["phase5c_reports"] / "final_fit_metadata.json").exists() else {}
    feature_names = list(final_meta.get("feature_names", []))
    return {
        "status": "valid" if not missing else "invalid",
        "missing_required_artifacts": missing,
        "phase5c_candidate_id": locked.get("candidate_id"),
        "phase5c_candidate_matches_expected": locked.get("candidate_id") == config["backbone"]["expected_candidate"],
        "phase5d_locked_candidate": phase5d_locked.get("candidate_id"),
        "phase5d_checkpoint_read_only": True,
        "phase5c_feature_schema_count": len(feature_names),
        "feature_schema_has_label_leakage": any(any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS) for name in feature_names),
        "backbone_frozen_required": bool(config["backbone"]["frozen"]),
        "transformer_training_called": TRANSFORMER_TRAINING_CALLED,
        "benchmark_labels_used_for_selection": False,
        "hard_failures": missing,
    }


def manifest_hash_map(manifest: list[dict[str, Any]]) -> dict[str, str]:
    return {str(row["artifact_key"]): str(row["sha256"]) for row in manifest if row.get("sha256")}


def source_hashes_unchanged(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> bool:
    return manifest_hash_map(before) == manifest_hash_map(after)


def read_phase5c_benchmark_features(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    allowed = [column for column in header if column not in BENCHMARK_LABEL_OR_ERROR_COLUMNS]
    return pd.read_csv(path, usecols=allowed)


def read_phase5c_benchmark_labels(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["subset", "global_engine_id", "true_rul"])


def strip_benchmark_labels_before_lock(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[column for column in BENCHMARK_LABEL_OR_ERROR_COLUMNS if column in frame.columns])


def gate_candidate_feature_names(config: dict[str, Any]) -> list[str]:
    names = [
        "base_rul_prediction",
        "lower_80",
        "lower_90",
        "interval_width_90",
        "normalized_interval_width",
        "high_error_risk_probability",
        "domain_support_score",
        "support_score",
        "operating_regime_distance",
        "regime_distance",
        "operating_regime_rarity",
        "transformer_health_score",
        "transformer_degradation_rate",
        "recent_base_rul_slope",
        "base_cycle_rate_residual",
        "base_monotonicity_residual",
        "sensor_change_norm",
        "valid_sequence_fraction",
        "padding_fraction",
    ]
    for sensor in config["features"]["retained_sensors"]:
        base = f"sensor_{int(sensor)}"
        names.extend([f"{base}_slope_5", f"{base}_slope_10", f"{base}_slope_gap", f"{base}_healthy_residual"])
    return list(dict.fromkeys(names))


def correction_candidate_feature_names(config: dict[str, Any]) -> list[str]:
    names = gate_candidate_feature_names(config) + [
        "gate_probability",
        "gate_threshold_margin",
        "gate_active_float",
        "correction_bound_proximity",
    ]
    return list(dict.fromkeys(names))


def audit_feature_leakage(feature_names: list[str]) -> dict[str, Any]:
    allowed_residual_features = {"base_cycle_rate_residual", "base_monotonicity_residual"}
    forbidden = [
        name
        for name in feature_names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
        or ("residual" in name.lower() and name not in allowed_residual_features and "healthy_residual" not in name.lower())
    ]
    return {"feature_count": len(feature_names), "forbidden_features": forbidden, "leakage_detected": bool(forbidden)}


def fit_selective_feature_preprocessor(
    frame: pd.DataFrame,
    config: dict[str, Any],
    candidate_names: list[str],
    *,
    maximum_features: int,
    feature_family: str,
) -> dict[str, Any]:
    monitoring_max = float(config["maintenance"]["monitoring_rul_max"])
    healthy = frame[frame["true_rul"].astype(float) > monitoring_max] if "true_rul" in frame else frame
    if healthy.empty:
        healthy = frame
    baselines: dict[str, float] = {}
    for sensor in config["features"]["retained_sensors"]:
        latest = f"sensor_{int(sensor)}_latest"
        baselines[latest] = float(healthy.get(latest, pd.Series(0.0)).astype(float).mean())

    if "true_rul" in frame:
        abs_residual = (frame["predicted_rul"].astype(float) - frame["true_rul"].astype(float)).abs()
        radius_80 = float(np.quantile(abs_residual, 0.80))
        radius_90 = float(np.quantile(abs_residual, 0.90))
    else:
        radius_80 = 15.0
        radius_90 = 25.0
    metadata = {"radius_80": radius_80, "radius_90": radius_90}
    augmented = augment_selective_features(frame, baselines, metadata, config)
    available = [name for name in candidate_names if name in augmented.columns and not audit_feature_leakage([name])["leakage_detected"]]
    essentials = [
        "base_rul_prediction",
        "lower_90",
        "normalized_interval_width",
        "high_error_risk_probability",
        "domain_support_score",
        "transformer_health_score",
        "recent_base_rul_slope",
        "sensor_change_norm",
    ]
    if feature_family == "correction":
        essentials.extend(["gate_probability", "gate_threshold_margin"])
    values = augmented[available].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    variances = values.var(axis=0).sort_values(ascending=False)
    selected: list[str] = []
    for name in essentials + [name for name in variances.index.tolist() if name not in essentials]:
        if name in available and name not in selected and (name in essentials or float(variances.get(name, 0.0)) > 1e-12):
            selected.append(name)
        if len(selected) >= int(maximum_features):
            break
    selected = sorted(selected, key=lambda item: available.index(item))
    selected_values = values[selected]
    mean = selected_values.mean(axis=0)
    std = selected_values.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return {
        "feature_family": feature_family,
        "feature_names": selected,
        "all_candidate_features": available,
        "healthy_baselines": baselines,
        "risk_metadata": metadata,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "input_clamp": float(config["features"]["input_clamp"]),
        "healthy_row_definition": f"true_rul > {monitoring_max}",
    }


def augment_selective_features(frame: pd.DataFrame, baselines: dict[str, float], risk_metadata: dict[str, float], config: dict[str, Any]) -> pd.DataFrame:
    result = add_healthy_residuals(frame, baselines, config)
    result["base_rul_prediction"] = result.get("base_rul_prediction", result["predicted_rul"]).astype(float)
    radius_80 = float(risk_metadata.get("radius_80", 15.0))
    radius_90 = float(risk_metadata.get("radius_90", 25.0))
    point = result["base_rul_prediction"].astype(float)
    result["lower_80"] = np.maximum(0.0, point - radius_80)
    result["lower_90"] = np.maximum(0.0, point - radius_90)
    result["interval_width_90"] = 2.0 * radius_90
    result["normalized_interval_width"] = result["interval_width_90"] / np.maximum(point, 1.0)
    support = result.get("domain_support_score", pd.Series(1.0, index=result.index)).astype(float).clip(0.0, 1.0)
    result["support_score"] = support
    regime_distance = result.get("operating_regime_distance", pd.Series(1.0 - support, index=result.index)).astype(float)
    result["regime_distance"] = regime_distance
    first_diff_columns = [f"sensor_{int(sensor)}_first_diff" for sensor in config["features"]["retained_sensors"] if f"sensor_{int(sensor)}_first_diff" in result.columns]
    if first_diff_columns:
        result["sensor_change_norm"] = np.sqrt(np.square(result[first_diff_columns].astype(float)).mean(axis=1))
    else:
        result["sensor_change_norm"] = 0.0
    cycle_rate = result.get("base_cycle_rate_residual", pd.Series(0.0, index=result.index)).astype(float)
    score = -1.5 + 0.03 * point + 0.55 * result["normalized_interval_width"].astype(float) + 1.4 * (1.0 - support) + 0.15 * cycle_rate
    result["high_error_risk_probability"] = 1.0 / (1.0 + np.exp(-np.clip(score, -30.0, 30.0)))
    if "gate_probability" not in result:
        result["gate_probability"] = 0.0
    if "gate_threshold" not in result:
        result["gate_threshold"] = 1.0
    result["gate_threshold_margin"] = result["gate_probability"].astype(float) - result["gate_threshold"].astype(float)
    result["gate_active_float"] = result.get("gate_active", pd.Series(False, index=result.index)).astype(float)
    if "correction_magnitude" in result:
        bound = max(float(result["correction_magnitude"].abs().max()), 1.0)
        result["correction_bound_proximity"] = result["correction_magnitude"].astype(float) / bound
    else:
        result["correction_bound_proximity"] = 0.0
    return result


def transform_selective_frame(frame: pd.DataFrame, preprocessor: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    augmented = augment_selective_features(frame, preprocessor["healthy_baselines"], preprocessor["risk_metadata"], config)
    selected = list(preprocessor["feature_names"])
    values = augmented[selected].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    mean = pd.Series(preprocessor["mean"])
    std = pd.Series(preprocessor["std"]).replace(0.0, 1.0)
    normalized = ((values - mean[selected]) / std[selected]).clip(-float(preprocessor["input_clamp"]), float(preprocessor["input_clamp"]))
    return normalized.to_numpy(dtype=np.float32), augmented


def dangerous_optimism_target(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    true = frame["true_rul"].astype(float).to_numpy()
    base = frame["predicted_rul"].astype(float).to_numpy()
    return ((true <= float(config["dangerous_optimism"]["true_rul_max"])) & ((base - true) >= float(config["dangerous_optimism"]["optimism_threshold"]))).astype(int)


def magnitude_target(frame: pd.DataFrame, config: dict[str, Any], bound: float) -> np.ndarray:
    dangerous = dangerous_optimism_target(frame, config).astype(bool)
    optimistic_amount = np.maximum(0.0, frame["predicted_rul"].astype(float).to_numpy() - frame["true_rul"].astype(float).to_numpy())
    return np.where(dangerous, np.minimum(float(bound), optimistic_amount), 0.0).astype(np.float32)


def add_dangerous_targets(frame: pd.DataFrame, config: dict[str, Any], bound: float) -> pd.DataFrame:
    result = frame.copy()
    result["dangerous_optimism"] = dangerous_optimism_target(result, config)
    result["optimism_amount"] = np.maximum(0.0, result["predicted_rul"].astype(float) - result["true_rul"].astype(float))
    result["magnitude_target"] = magnitude_target(result, config, bound)
    return result


def dangerous_definition_report(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for cutoff in config["dangerous_optimism"]["sensitivity_rul_cutoffs"]:
        for threshold in config["dangerous_optimism"]["sensitivity_optimism_thresholds"]:
            true = frame["true_rul"].astype(float)
            base = frame["predicted_rul"].astype(float)
            label = (true <= float(cutoff)) & ((base - true) >= float(threshold))
            rows.append({"true_rul_max": float(cutoff), "optimism_threshold": float(threshold), "event_count": int(label.sum()), "event_prevalence": float(label.mean())})
    primary = dangerous_optimism_target(frame, config)
    return {
        "definition_id": "true_rul_le_30_and_base_minus_true_ge_10",
        "true_rul_max": float(config["dangerous_optimism"]["true_rul_max"]),
        "optimism_threshold": float(config["dangerous_optimism"]["optimism_threshold"]),
        "locked_before_candidate_comparison": True,
        "selected_using_benchmark": False,
        "event_count": int(primary.sum()),
        "event_prevalence": float(primary.mean()),
        "sensitivity": rows,
    }


def safe_row_zero_correction_loss(predicted_magnitude: torch.Tensor, dangerous: torch.Tensor) -> torch.Tensor:
    return ((1.0 - dangerous.float()) * predicted_magnitude.pow(2)).mean()


def critical_under_correction_loss(predicted_magnitude: torch.Tensor, target_magnitude: torch.Tensor, critical_weight: float) -> torch.Tensor:
    return float(critical_weight) * torch.clamp(target_magnitude - predicted_magnitude, min=0.0).pow(2).mean()


def one_sided_final_prediction(base_rul: np.ndarray, gate_probability: np.ndarray, magnitude: np.ndarray, *, threshold: float, bound: float, hard_gate: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(base_rul, dtype=float)
    probability = np.clip(np.asarray(gate_probability, dtype=float), 0.0, 1.0)
    bounded_magnitude = np.clip(np.asarray(magnitude, dtype=float), 0.0, float(bound))
    if hard_gate:
        gate_weight = (probability >= float(threshold)).astype(float)
    else:
        gate_weight = probability
    downward = np.clip(gate_weight * bounded_magnitude, 0.0, float(bound))
    final = np.maximum(0.0, base - downward)
    final = np.minimum(final, base)
    inactive = gate_weight <= 0.0
    final[inactive] = base[inactive]
    return final.astype(float), downward.astype(float), gate_weight.astype(float)


def verify_one_sided_property(frame: pd.DataFrame, *, atol: float = 1e-8) -> dict[str, Any]:
    base = frame["base_predicted_rul"].astype(float).to_numpy()
    final = frame["corrected_predicted_rul"].astype(float).to_numpy()
    inactive = ~frame["gate_active"].astype(bool).to_numpy() if "gate_active" in frame else np.zeros(len(frame), dtype=bool)
    return {
        "row_count": int(len(frame)),
        "never_exceeds_base": bool(np.all(final <= base + atol)),
        "inactive_exact_fallback": bool(np.allclose(final[inactive], base[inactive], atol=atol, rtol=0.0)) if inactive.any() else True,
        "final_nonnegative": bool(np.all(final >= -atol)),
        "maximum_positive_delta": float(np.max(final - base)) if len(frame) else 0.0,
    }


def select_gate_threshold(y_true: np.ndarray, probability: np.ndarray, config: dict[str, Any], critical_mask: np.ndarray | None = None) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(p) == 0:
        return 1.0
    candidates = np.unique(np.concatenate(([0.0, 1.0], p, np.quantile(p, np.linspace(0.0, 1.0, 101)))))
    max_activation = float(config["gate"]["activation_rate_max"])
    min_recall = float(config["gate"]["minimum_dangerous_recall"])
    min_critical = float(config["gate"]["minimum_critical_dangerous_recall"])
    rows = []
    crit = np.asarray(critical_mask, dtype=bool) if critical_mask is not None else np.zeros(len(y), dtype=bool)
    for threshold in candidates:
        active = p >= float(threshold)
        recall = float((active & (y == 1)).sum() / max((y == 1).sum(), 1))
        critical_positive = (y == 1) & crit
        critical_recall = float((active & critical_positive).sum() / max(critical_positive.sum(), 1))
        activation = float(active.mean())
        precision = float((active & (y == 1)).sum() / max(active.sum(), 1))
        feasible = activation <= max_activation + 1e-12 and recall >= min_recall and critical_recall >= min_critical
        rows.append((feasible, recall, critical_recall, -activation, precision, float(threshold)))
    feasible_rows = [row for row in rows if row[0]]
    if feasible_rows:
        return float(sorted(feasible_rows, reverse=True)[0][-1])
    under_activation = [row for row in rows if -row[3] <= max_activation + 1e-12]
    if under_activation:
        return float(sorted(under_activation, reverse=True)[0][-1])
    return float(np.quantile(p, max(0.0, 1.0 - max_activation)))


def calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (p >= start) & (p < end if end < 1.0 else p <= end)
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)


def safe_roc_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, probability))


def safe_average_precision(y_true: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, probability))


def gate_metrics(frame: pd.DataFrame, probability: np.ndarray, threshold: float, config: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    y = dangerous_optimism_target(frame, config)
    active = np.asarray(probability, dtype=float) >= float(threshold)
    positive = y == 1
    negative = ~positive
    critical_positive = positive & (frame["true_rul"].astype(float).to_numpy() <= float(config["safety_loss"]["critical_rul_max"]))
    subgroup_recalls = []
    for _, group in frame.assign(_y=y, _active=active).groupby("subset", observed=False):
        pos = group["_y"].to_numpy(dtype=int) == 1
        if pos.sum() > 0:
            subgroup_recalls.append(float((group["_active"].to_numpy(dtype=bool) & pos).sum() / pos.sum()))
    supported_recalls = []
    for _, group in frame.assign(_y=y, _active=active).groupby("operating_regime", observed=False):
        pos = group["_y"].to_numpy(dtype=int) == 1
        if pos.sum() >= 5:
            supported_recalls.append(float((group["_active"].to_numpy(dtype=bool) & pos).sum() / pos.sum()))
    prevalence = float(positive.mean()) if len(positive) else 0.0
    auprc = safe_average_precision(y, probability)
    return {
        "candidate_id": candidate_id,
        "row_count": int(len(frame)),
        "event_count": int(positive.sum()),
        "event_prevalence": prevalence,
        "threshold": float(threshold),
        "auroc": safe_roc_auc(y, probability),
        "auprc": auprc,
        "brier_score": float(brier_score_loss(y, np.clip(probability, 0.0, 1.0))) if len(np.unique(y)) > 1 else 0.0,
        "calibration_error": calibration_error(y, probability),
        "dangerous_event_recall": float((active & positive).sum() / max(positive.sum(), 1)),
        "dangerous_event_precision": float((active & positive).sum() / max(active.sum(), 1)),
        "false_positive_rate": float((active & negative).sum() / max(negative.sum(), 1)),
        "gate_activation_rate": float(active.mean()) if len(active) else 0.0,
        "critical_dangerous_event_recall": float((active & critical_positive).sum() / max(critical_positive.sum(), 1)),
        "worst_split_recall": float(min(subgroup_recalls)) if subgroup_recalls else 0.0,
        "worst_supported_subgroup_recall": float(min(supported_recalls)) if supported_recalls else 0.0,
        "auprc_above_prevalence": bool(auprc > prevalence),
    }


def fit_sparse_kan_gate(model: SparseKANGateModel, train_x: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> SparseKANGateModel:
    x = torch.as_tensor(train_x, dtype=torch.float32)
    target = torch.as_tensor(y.astype(np.float32), dtype=torch.float32)
    pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["gate_learning_rate"]))
    batch_size = min(int(config["training"]["batch_size"]), len(x))
    generator = torch.Generator().manual_seed(int(config["training"]["random_seed"]))
    for _ in range(int(config["gate"].get("sparse_kan_epochs", 2))):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            idx = order[start : start + batch_size]
            logits = model(x[idx])
            loss = F.binary_cross_entropy_with_logits(logits, target[idx], pos_weight=pos_weight)
            loss = loss + 0.001 * edge_sparsity_penalty(model) + 0.0001 * spline_smoothness_penalty(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def fit_gate_candidate(candidate_id: str, train: pd.DataFrame, config: dict[str, Any]) -> GateFit:
    preprocessor = fit_selective_feature_preprocessor(
        train,
        config,
        gate_candidate_feature_names(config),
        maximum_features=int(config["gate"]["maximum_features"]),
        feature_family="gate",
    )
    x, augmented = transform_selective_frame(train, preprocessor, config)
    y = dangerous_optimism_target(train, config)
    critical = train["true_rul"].astype(float).to_numpy() <= float(config["safety_loss"]["critical_rul_max"])
    if candidate_id == "no_gate":
        model: Any = NoGateModel()
        probability = model.predict_proba(x)
    elif len(np.unique(y)) < 2:
        model = ConstantProbabilityGateModel(float(np.mean(y)))
        probability = model.predict_proba(x)[:, 1]
    elif candidate_id == "rule":
        weights = {
            "high_error_risk_probability": 1.0,
            "normalized_interval_width": 0.4,
            "regime_distance": 0.25,
            "base_cycle_rate_residual": 0.1,
            "base_rul_prediction": -0.005,
        }
        model = RuleGateModel(weights, 0.0, 1.0)
        raw = model.score_frame(augmented)
        model = RuleGateModel(weights, float(np.min(raw)), float(np.max(raw)))
        probability = model.predict_proba_from_frame(augmented)
    elif candidate_id in {"logistic", "isotonic_logistic"}:
        logistic = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=int(config["training"]["random_seed"]))
        logistic.fit(x, y)
        if candidate_id == "isotonic_logistic":
            base_probability = logistic.predict_proba(x)[:, 1]
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(base_probability, y)
            model = IsotonicLogisticGateModel(logistic, calibrator)
            probability = model.predict_proba(x)
        else:
            model = logistic
            probability = logistic.predict_proba(x)[:, 1]
    elif candidate_id == "shallow_tree":
        model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=int(config["gate"]["shallow_tree_min_leaf"]), class_weight="balanced", random_state=int(config["training"]["random_seed"]))
        model.fit(x, y)
        probability = model.predict_proba(x)[:, 1]
    elif candidate_id == "sparse_additive_kan":
        model = SparseKANGateModel(x.shape[1], grid_size=5, spline_degree=int(config["correction"]["spline_degree"]), input_clamp=float(config["features"]["input_clamp"]), seed=int(config["training"]["random_seed"]))
        model = fit_sparse_kan_gate(model, x, y, config)
        with torch.no_grad():
            probability = model.probability(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
    else:
        raise ValueError(f"Unknown gate candidate: {candidate_id}")
    threshold = 0.0 if candidate_id == "no_gate" else select_gate_threshold(y, probability, config, critical)
    metrics = gate_metrics(train, probability, threshold, config, candidate_id)
    return GateFit({"candidate_id": candidate_id}, model, preprocessor, threshold, metrics)


def predict_gate_probability(fit: GateFit, frame: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    x, augmented = transform_selective_frame(frame, fit.preprocessor, config)
    candidate_id = str(fit.candidate["candidate_id"])
    if isinstance(fit.model, ConstantProbabilityGateModel):
        probability = fit.model.predict_proba(x)[:, 1]
    elif candidate_id == "no_gate":
        probability = np.ones(len(frame), dtype=float)
    elif candidate_id == "rule":
        probability = fit.model.predict_proba_from_frame(augmented)
    elif candidate_id in {"logistic", "shallow_tree"}:
        raw_probability = fit.model.predict_proba(x)
        probability = raw_probability[:, 1] if raw_probability.ndim == 2 else raw_probability
    elif candidate_id == "isotonic_logistic":
        probability = fit.model.predict_proba(x)
    elif candidate_id == "sparse_additive_kan":
        with torch.no_grad():
            probability = fit.model.probability(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
    else:
        raise ValueError(f"Unknown gate candidate: {candidate_id}")
    return np.clip(probability.astype(float), 0.0, 1.0), augmented


def add_gate_columns(frame: pd.DataFrame, fit: GateFit, config: dict[str, Any]) -> pd.DataFrame:
    probability, _ = predict_gate_probability(fit, frame, config)
    result = frame.copy()
    result["gate_probability"] = probability
    result["gate_threshold"] = float(fit.threshold)
    result["gate_active"] = probability >= float(fit.threshold)
    result["gate_threshold_margin"] = result["gate_probability"].astype(float) - float(fit.threshold)
    result["gate_active_float"] = result["gate_active"].astype(float)
    return result


def screen_risk_gate_candidates(selection_frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, GateFit, dict[str, Any]]:
    dev, val, split = split_by_engine(selection_frame, float(config["selection"]["development_fraction"]), int(config["selection"]["screening_seed"]))
    rows = []
    calibration_rows = []
    fits: list[GateFit] = []
    for candidate_id in ["no_gate"] + [candidate for candidate in config["gate"]["candidates"] if candidate != "no_gate"]:
        fit = fit_gate_candidate(candidate_id, dev, config)
        probability, _ = predict_gate_probability(fit, val, config)
        metrics = gate_metrics(val, probability, fit.threshold, config, candidate_id)
        metrics["selected_threshold_from_development"] = fit.threshold
        rows.append(metrics)
        fits.append(fit)
        bins = np.linspace(0.0, 1.0, 11)
        y = dangerous_optimism_target(val, config)
        for index, (start, end) in enumerate(zip(bins[:-1], bins[1:]), start=1):
            mask = (probability >= start) & (probability < end if end < 1.0 else probability <= end)
            if mask.any():
                calibration_rows.append({"candidate_id": candidate_id, "bin": index, "probability_mean": float(probability[mask].mean()), "event_rate": float(y[mask].mean()), "row_count": int(mask.sum())})
    metrics_frame = pd.DataFrame(rows)
    metrics_frame["eligible"] = (
        (metrics_frame["dangerous_event_recall"] >= float(config["gate"]["minimum_dangerous_recall"]))
        & (metrics_frame["critical_dangerous_event_recall"] >= float(config["gate"]["minimum_critical_dangerous_recall"]))
        & (metrics_frame["gate_activation_rate"] <= float(config["gate"]["activation_rate_max"]))
        & (metrics_frame["auprc"] > metrics_frame["event_prevalence"])
    )
    ranked = metrics_frame.sort_values(["eligible", "dangerous_event_recall", "critical_dangerous_event_recall", "gate_activation_rate", "auprc"], ascending=[False, False, False, True, False], kind="mergesort")
    selected_id = str(ranked.iloc[0]["candidate_id"])
    if not bool(ranked.iloc[0]["eligible"]):
        under_activation = ranked[ranked["gate_activation_rate"] <= float(config["gate"]["activation_rate_max"])]
        if not under_activation.empty:
            selected_id = str(under_activation.sort_values(["dangerous_event_recall", "critical_dangerous_event_recall", "auprc"], ascending=[False, False, False], kind="mergesort").iloc[0]["candidate_id"])
    selected_fit = next(fit for fit in fits if fit.candidate["candidate_id"] == selected_id)
    selected_fit.metrics = ranked[ranked["candidate_id"] == selected_id].iloc[0].to_dict()
    return metrics_frame, pd.DataFrame(calibration_rows), selected_fit, split


def build_correction_candidate_registry(config: dict[str, Any], *, smoke: bool = False) -> list[dict[str, Any]]:
    seed = int(config["training"]["random_seed"])
    bounds = [float(value) for value in config["correction"]["bounds"]]
    if smoke:
        bounds = [min(bounds), max(bounds)]
    candidates: list[dict[str, Any]] = [{"candidate_id": "phase5c_exact_fallback", "candidate_type": "baseline", "correction_bound": 0.0}]
    for bound in bounds:
        suffix = str(int(bound))
        candidates.extend(
            [
                {"candidate_id": f"constant_downward_bound{suffix}", "candidate_type": "constant", "correction_bound": bound},
                {"candidate_id": f"linear_nonnegative_bound{suffix}", "candidate_type": "linear_nonnegative", "correction_bound": bound},
                {"candidate_id": f"mlp_magnitude_bound{suffix}", "candidate_type": "mlp_magnitude", "correction_bound": bound, "hidden_dim": int(config["training"]["mlp_hidden"]), "seed": seed + int(bound)},
                {"candidate_id": f"single_layer_sparse_additive_kan_bound{suffix}", "candidate_type": "one_sided_kan", "correction_bound": bound, "grid_size": 5, "spline_degree": int(config["correction"]["spline_degree"]), "hidden_nodes": 0, "zero_weight": 1.0, "critical_weight": 4.0, "sparsity": 0.001, "smoothness": 0.0001, "seed": seed + 100 + int(bound)},
                {"candidate_id": f"safety_weighted_sparse_additive_kan_bound{suffix}", "candidate_type": "one_sided_kan", "correction_bound": bound, "grid_size": 5, "spline_degree": int(config["correction"]["spline_degree"]), "hidden_nodes": 0, "zero_weight": 2.0, "critical_weight": 8.0, "sparsity": 0.001, "smoothness": 0.0001, "seed": seed + 200 + int(bound)},
                {"candidate_id": f"regime_shrinkage_additive_kan_bound{suffix}", "candidate_type": "one_sided_kan", "correction_bound": bound, "grid_size": 7, "spline_degree": int(config["correction"]["spline_degree"]), "hidden_nodes": 0, "zero_weight": 2.0, "critical_weight": 4.0, "sparsity": 0.002, "smoothness": 0.001, "seed": seed + 300 + int(bound)},
            ]
        )
    candidates.append({"candidate_id": "phase5d_two_layer_one_sided_control_bound20", "candidate_type": "one_sided_kan", "correction_bound": 20.0, "grid_size": 5, "spline_degree": int(config["correction"]["spline_degree"]), "hidden_nodes": 8, "zero_weight": 1.0, "critical_weight": 4.0, "sparsity": 0.001, "smoothness": 0.001, "seed": seed + 400})
    return candidates


def train_torch_magnitude_model(
    model: nn.Module,
    candidate: dict[str, Any],
    train_x: np.ndarray,
    target: np.ndarray,
    dangerous: np.ndarray,
    weights: np.ndarray,
    config: dict[str, Any],
    epochs: int,
) -> nn.Module:
    torch.manual_seed(int(candidate.get("seed", config["training"]["random_seed"])))
    x = torch.as_tensor(train_x, dtype=torch.float32)
    y = torch.as_tensor(target.astype(np.float32), dtype=torch.float32)
    d = torch.as_tensor(dangerous.astype(np.float32), dtype=torch.float32)
    w = torch.as_tensor(weights.astype(np.float32), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["correction_learning_rate"]))
    batch_size = min(int(config["training"]["batch_size"]), len(x))
    generator = torch.Generator().manual_seed(int(candidate.get("seed", config["training"]["random_seed"])))
    for _ in range(int(epochs)):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            idx = order[start : start + batch_size]
            pred = model(x[idx])
            magnitude_loss = F.huber_loss(pred, y[idx], reduction="none")
            loss = (w[idx] * magnitude_loss).mean()
            loss = loss + critical_under_correction_loss(pred, y[idx], float(candidate.get("critical_weight", 4.0))) * 0.01
            loss = loss + float(candidate.get("zero_weight", 1.0)) * safe_row_zero_correction_loss(pred, d[idx]) * 0.01
            loss = loss + float(config["correction"]["intervention_penalty"]) * pred.mean()
            loss = loss + float(candidate.get("sparsity", 0.0)) * edge_sparsity_penalty(model)
            loss = loss + float(candidate.get("smoothness", 0.0)) * spline_smoothness_penalty(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def fit_correction_candidate(candidate: dict[str, Any], train: pd.DataFrame, config: dict[str, Any], epochs: int) -> CorrectionFit:
    preprocessor = fit_selective_feature_preprocessor(
        train,
        config,
        correction_candidate_feature_names(config),
        maximum_features=int(config["features"]["maximum_correction_features"]),
        feature_family="correction",
    )
    x, _ = transform_selective_frame(train, preprocessor, config)
    bound = float(candidate.get("correction_bound", 0.0))
    target = magnitude_target(train, config, bound)
    dangerous = dangerous_optimism_target(train, config)
    weights = engine_balanced_weights(train)
    if candidate["candidate_type"] == "baseline":
        model: Any = ConstantMagnitudeModel(0.0, 0.0)
    elif candidate["candidate_type"] == "constant":
        positive = target[target > 0]
        magnitude = float(np.median(positive)) if len(positive) else min(bound, float(config["dangerous_optimism"]["optimism_threshold"]))
        model = ConstantMagnitudeModel(magnitude, bound)
    elif candidate["candidate_type"] == "linear_nonnegative":
        regressor = LinearRegression(positive=True)
        regressor.fit(x, target, sample_weight=weights)
        model = LinearNonNegativeMagnitudeModel(regressor, bound)
    elif candidate["candidate_type"] == "mlp_magnitude":
        model = MLPMagnitudeModel(x.shape[1], int(candidate["hidden_dim"]), bound, int(candidate["seed"]))
        model = train_torch_magnitude_model(model, candidate, x, target, dangerous, weights, config, epochs)
    elif candidate["candidate_type"] == "one_sided_kan":
        model = OneSidedKANMagnitude(
            x.shape[1],
            correction_bound=bound,
            grid_size=int(candidate["grid_size"]),
            spline_degree=int(candidate["spline_degree"]),
            input_clamp=float(config["features"]["input_clamp"]),
            hidden_nodes=int(candidate.get("hidden_nodes", 0)),
            seed=int(candidate["seed"]),
        )
        model = train_torch_magnitude_model(model, candidate, x, target, dangerous, weights, config, epochs)
    else:
        raise ValueError(f"Unknown correction candidate type: {candidate['candidate_type']}")
    return CorrectionFit(candidate, model, preprocessor, {})


def predict_correction_magnitude(fit: CorrectionFit, frame: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    x, augmented = transform_selective_frame(frame, fit.preprocessor, config)
    if fit.candidate["candidate_type"] in {"baseline", "constant", "linear_nonnegative"}:
        magnitude = fit.model.predict(x)
    elif fit.candidate["candidate_type"] in {"mlp_magnitude", "one_sided_kan"}:
        with torch.no_grad():
            magnitude = fit.model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
    else:
        raise ValueError(f"Unknown correction candidate type: {fit.candidate['candidate_type']}")
    return np.clip(magnitude.astype(float), 0.0, float(fit.candidate.get("correction_bound", 0.0))), augmented


def selective_corrected_predictions(frame: pd.DataFrame, gate_fit: GateFit, correction_fit: CorrectionFit, config: dict[str, Any]) -> pd.DataFrame:
    with_gate = add_gate_columns(frame, gate_fit, config)
    magnitude, _ = predict_correction_magnitude(correction_fit, with_gate, config)
    final, downward, gate_weight = one_sided_final_prediction(
        with_gate["predicted_rul"].astype(float).to_numpy(),
        with_gate["gate_probability"].astype(float).to_numpy(),
        magnitude,
        threshold=float(gate_fit.threshold),
        bound=float(correction_fit.candidate.get("correction_bound", 0.0)),
        hard_gate=bool(config["gate"].get("hard_threshold", True)),
    )
    result = with_gate.copy()
    result["base_predicted_rul"] = result["predicted_rul"].astype(float)
    result["correction_magnitude"] = magnitude
    result["downward_correction"] = downward
    result["kan_correction"] = -downward
    result["gate_weight"] = gate_weight
    result["gate_active"] = gate_weight > 0.0
    result["corrected_predicted_rul"] = final
    if "true_rul" in result:
        result["corrected_residual"] = result["corrected_predicted_rul"].astype(float) - result["true_rul"].astype(float)
        result["corrected_absolute_error"] = result["corrected_residual"].abs()
        result["corrected_squared_error"] = result["corrected_residual"] ** 2
    check = verify_one_sided_property(result)
    if not (check["never_exceeds_base"] and check["inactive_exact_fallback"] and check["final_nonnegative"]):
        raise AssertionError(f"Selective one-sided safety property failed: {check}")
    return result


def selective_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    metrics = point_metrics(frame, frame["corrected_predicted_rul"].to_numpy(dtype=float), correction=frame["kan_correction"].to_numpy(dtype=float), policy_threshold=float(config["maintenance"]["critical_rul_max"]))
    base_metrics = point_metrics(frame, frame["base_predicted_rul"].to_numpy(dtype=float), correction=np.zeros(len(frame)), policy_threshold=float(config["maintenance"]["critical_rul_max"]))
    y_true = frame["true_rul"].astype(float).to_numpy()
    base = frame["base_predicted_rul"].astype(float).to_numpy()
    corrected = frame["corrected_predicted_rul"].astype(float).to_numpy()
    downward = frame["downward_correction"].astype(float).to_numpy()
    active = frame["gate_active"].astype(bool).to_numpy()
    critical = y_true <= float(config["maintenance"]["critical_rul_max"])
    noncritical = ~critical
    safe = y_true > float(config["maintenance"]["inspection_rul_max"])
    base_missed = critical & (base > float(config["maintenance"]["critical_rul_max"]))
    corrected_missed = critical & (corrected > float(config["maintenance"]["critical_rul_max"]))
    metrics.update(
        {
            "base_mae": base_metrics["mae"],
            "base_rmse": base_metrics["rmse"],
            "base_nasa_score": base_metrics["nasa_score"],
            "base_critical_miss_proxy_count": base_metrics["critical_miss_proxy_count"],
            "gate_activation_count": int(active.sum()),
            "gate_activation_rate": float(active.mean()) if len(active) else 0.0,
            "unchanged_count": int(np.isclose(base, corrected, atol=1e-10, rtol=0.0).sum()),
            "unchanged_rate": float(np.isclose(base, corrected, atol=1e-10, rtol=0.0).mean()) if len(active) else 0.0,
            "mean_downward_correction": float(downward.mean()) if len(downward) else 0.0,
            "mean_activated_correction": float(downward[active].mean()) if active.any() else 0.0,
            "p90_downward_correction": float(np.quantile(downward, 0.90)) if len(downward) else 0.0,
            "bound_saturation_rate": float(np.mean(downward >= 0.98 * max(frame["correction_magnitude"].max(), 1e-12))) if len(downward) else 0.0,
            "noncritical_unnecessary_correction_count": int((noncritical & active).sum()),
            "noncritical_unnecessary_correction_rate": float((noncritical & active).sum() / max(noncritical.sum(), 1)),
            "safe_row_unnecessary_correction_count": int((safe & active).sum()),
            "safe_row_unnecessary_correction_rate": float((safe & active).sum() / max(safe.sum(), 1)),
            "correction_benefit_rate": float((np.abs(corrected - y_true) < np.abs(base - y_true)).mean()) if len(y_true) else 0.0,
            "correction_harm_rate": float((np.abs(corrected - y_true) > np.abs(base - y_true)).mean()) if len(y_true) else 0.0,
            "previous_misses_corrected": int((base_missed & ~corrected_missed).sum()),
            "previous_misses_still_missed": int((base_missed & corrected_missed).sum()),
            "new_critical_misses": int((~base_missed & corrected_missed).sum()),
            "critical_gate_failed_count": int((critical & ~active).sum()),
            "critical_insufficient_correction_count": int((critical & active & corrected_missed).sum()),
        }
    )
    return metrics


def selection_flags(metrics: dict[str, Any], base_metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    miss_improves = metrics["critical_miss_proxy_count"] < base_metrics["critical_miss_proxy_count"] if base_metrics["critical_miss_proxy_count"] > 0 else metrics["critical_miss_proxy_count"] <= base_metrics["critical_miss_proxy_count"]
    safety = (
        miss_improves
        and metrics["critical_optimistic_rate"] <= base_metrics["critical_optimistic_rate"] + 1e-12
        and metrics["severe_optimistic_rate"] <= base_metrics["severe_optimistic_rate"] + float(config["selection"]["severe_optimistic_tolerance"])
        and (metrics["new_critical_misses"] == 0 if bool(config["selection"].get("no_new_critical_misses", True)) else True)
    )
    accuracy = (
        metrics["rmse"] <= base_metrics["rmse"] + float(config["selection"]["rmse_noninferiority_margin"])
        and metrics["mae"] <= base_metrics["mae"] + float(config["selection"]["mae_noninferiority_margin"])
        and metrics["nasa_score"] <= base_metrics["nasa_score"] * (1.0 + float(config["selection"]["nasa_material_worsening_fraction"]))
    )
    efficiency = metrics["gate_activation_rate"] <= float(config["selection"]["maximum_gate_activation"]) and metrics["bound_saturation_rate"] <= float(config["selection"]["maximum_bound_saturation"])
    return {"stage1_safety": bool(safety), "stage2_accuracy": bool(accuracy), "stage3_efficiency": bool(efficiency), "eligible": bool(safety and accuracy)}


def screen_correction_candidates(selection_frame: pd.DataFrame, gate_fit: GateFit, candidates: list[dict[str, Any]], config: dict[str, Any]) -> tuple[pd.DataFrame, list[CorrectionFit], dict[str, Any]]:
    dev, val, split = split_by_engine(selection_frame, float(config["selection"]["development_fraction"]), int(config["selection"]["screening_seed"]) + 17)
    local_gate = fit_gate_candidate(str(gate_fit.candidate["candidate_id"]), dev, config)
    dev_gate = add_gate_columns(dev, local_gate, config)
    val_gate = add_gate_columns(val, local_gate, config)
    base_metrics = selective_metrics(selective_corrected_predictions(val, GateFit({"candidate_id": "no_gate"}, NoGateModel(), local_gate.preprocessor, 2.0, {}), CorrectionFit({"candidate_id": "phase5c_exact_fallback", "candidate_type": "baseline", "correction_bound": 0.0}, ConstantMagnitudeModel(0.0, 0.0), fit_selective_feature_preprocessor(val_gate, config, correction_candidate_feature_names(config), maximum_features=int(config["features"]["maximum_correction_features"]), feature_family="correction"), {}), config), config)
    rows = []
    fits: list[CorrectionFit] = []
    for candidate in candidates:
        fit = fit_correction_candidate(candidate, dev_gate, config, int(config["training"]["screening_epochs"]))
        corrected = selective_corrected_predictions(val_gate, local_gate, fit, config)
        metrics = selective_metrics(corrected, config)
        flags = selection_flags(metrics, base_metrics, config)
        metrics.update(flags)
        metrics.update({"candidate_id": candidate["candidate_id"], "candidate_type": candidate["candidate_type"], "correction_bound": float(candidate.get("correction_bound", 0.0))})
        rows.append(metrics)
        fit.metrics = metrics
        fits.append(fit)
    frame = pd.DataFrame(rows)
    return frame, fits, split


def select_correction_finalists(screening: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    ranked = screening.sort_values(
        ["eligible", "stage1_safety", "stage2_accuracy", "critical_miss_proxy_count", "rmse", "gate_activation_rate", "mean_activated_correction"],
        ascending=[False, False, False, True, True, True, True],
        kind="mergesort",
    )
    finalist_count = int(config["selection"]["finalist_count"])
    finalists = ["phase5c_exact_fallback"]
    for candidate_id in ranked["candidate_id"].tolist():
        if candidate_id not in finalists:
            finalists.append(str(candidate_id))
        if len(finalists) >= finalist_count:
            break
    kan_candidates = ranked[(ranked["candidate_type"] == "one_sided_kan") & (~ranked["candidate_id"].isin(finalists))]
    if not any("kan" in item for item in finalists) and not kan_candidates.empty:
        finalists[-1] = str(kan_candidates.iloc[0]["candidate_id"])
    return finalists


def run_finalist_cross_validation(selection_frame: pd.DataFrame, gate_family: str, candidates: list[dict[str, Any]], finalist_ids: list[str], config: dict[str, Any]) -> pd.DataFrame:
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    rows = []
    for seed in config["selection"]["seeds"]:
        for train, val, split in kfold_engine_splits(selection_frame, int(config["selection"]["folds"]), int(seed)):
            gate_fit = fit_gate_candidate(gate_family, train, config)
            train_gate = add_gate_columns(train, gate_fit, config)
            val_gate = add_gate_columns(val, gate_fit, config)
            baseline_fit = CorrectionFit({"candidate_id": "phase5c_exact_fallback", "candidate_type": "baseline", "correction_bound": 0.0}, ConstantMagnitudeModel(0.0, 0.0), fit_selective_feature_preprocessor(train_gate, config, correction_candidate_feature_names(config), maximum_features=int(config["features"]["maximum_correction_features"]), feature_family="correction"), {})
            baseline_metrics = selective_metrics(selective_corrected_predictions(val_gate, gate_fit, baseline_fit, config), config)
            for candidate_id in finalist_ids:
                fit = fit_correction_candidate(by_id[candidate_id], train_gate, config, int(config["training"]["finalist_epochs"]))
                corrected = selective_corrected_predictions(val_gate, gate_fit, fit, config)
                metrics = selective_metrics(corrected, config)
                metrics.update(selection_flags(metrics, baseline_metrics, config))
                metrics.update({"candidate_id": candidate_id, "candidate_type": by_id[candidate_id]["candidate_type"], **split})
                rows.append(metrics)
    return pd.DataFrame(rows)


def choose_locked_correction(cv: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    aggregate = cv.groupby(["candidate_id", "candidate_type"], observed=False).agg(
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        nasa_score=("nasa_score", "mean"),
        severe_optimistic_rate=("severe_optimistic_rate", "mean"),
        critical_optimistic_rate=("critical_optimistic_rate", "mean"),
        critical_miss_proxy_count=("critical_miss_proxy_count", "mean"),
        new_critical_misses=("new_critical_misses", "mean"),
        gate_activation_rate=("gate_activation_rate", "mean"),
        unchanged_rate=("unchanged_rate", "mean"),
        bound_saturation_rate=("bound_saturation_rate", "mean"),
        stage1_safety=("stage1_safety", "mean"),
        stage2_accuracy=("stage2_accuracy", "mean"),
        eligible=("eligible", "mean"),
    ).reset_index()
    aggregate["eligible_all_folds"] = aggregate["eligible"] >= 1.0
    aggregate["stage1_all_folds"] = aggregate["stage1_safety"] >= 1.0
    aggregate["stage2_all_folds"] = aggregate["stage2_accuracy"] >= 1.0
    ranked = aggregate.sort_values(
        ["eligible_all_folds", "stage1_all_folds", "stage2_all_folds", "critical_miss_proxy_count", "rmse", "gate_activation_rate", "unchanged_rate"],
        ascending=[False, False, False, True, True, True, False],
        kind="mergesort",
    )
    selected = ranked.iloc[0].to_dict()
    if not bool(selected["eligible_all_folds"]):
        baseline = ranked[ranked["candidate_id"] == "phase5c_exact_fallback"]
        selected = baseline.iloc[0].to_dict() if not baseline.empty else selected
        selected["selective_kan_unsuccessful"] = True
    else:
        selected["selective_kan_unsuccessful"] = False
    selected["candidate_rank_table"] = ranked.to_dict("records")
    return selected


def pruning_decision(correlation: float, mean_abs_delta: float, no_new_misses: bool, config: dict[str, Any]) -> dict[str, Any]:
    accepted = (
        float(correlation) >= float(config["correction"]["pruning_correlation_min"])
        and float(mean_abs_delta) <= float(config["correction"]["pruning_mean_abs_delta_max"])
        and bool(no_new_misses)
    )
    return {"accepted": bool(accepted), "correlation": float(correlation), "mean_absolute_prediction_delta": float(mean_abs_delta), "no_new_critical_misses": bool(no_new_misses)}


def prune_if_accepted(correction_fit: CorrectionFit, validation_frame: pd.DataFrame, gate_fit: GateFit, config: dict[str, Any]) -> tuple[CorrectionFit, dict[str, Any]]:
    if correction_fit.candidate["candidate_type"] != "one_sided_kan":
        return correction_fit, {"applied": False, "accepted": False, "edges_before": 0, "edges_after": 0, "reason": "not_a_kan_model"}
    before = selective_corrected_predictions(validation_frame, gate_fit, correction_fit, config)
    pruned_model = copy.deepcopy(correction_fit.model)
    reports = [prune_layer_by_quantile(layer, float(config["correction"]["pruning_quantile"])) for layer in collect_kan_layers(pruned_model)]
    pruned_fit = CorrectionFit(correction_fit.candidate, pruned_model, correction_fit.preprocessor, correction_fit.metrics)
    after = selective_corrected_predictions(validation_frame, gate_fit, pruned_fit, config)
    before_pred = before["corrected_predicted_rul"].to_numpy(dtype=float)
    after_pred = after["corrected_predicted_rul"].to_numpy(dtype=float)
    correlation = float(np.corrcoef(before_pred, after_pred)[0, 1]) if len(before_pred) > 1 and np.std(after_pred) > 0 else 1.0
    mean_abs_delta = float(np.mean(np.abs(before_pred - after_pred)))
    before_metrics = selective_metrics(before, config)
    after_metrics = selective_metrics(after, config)
    no_new_misses = after_metrics["critical_miss_proxy_count"] <= before_metrics["critical_miss_proxy_count"]
    decision = pruning_decision(correlation, mean_abs_delta, no_new_misses, config)
    edges_before = int(sum(report.edges_before for report in reports))
    edges_after = int(sum(report.edges_after for report in reports))
    report = {
        "applied": True,
        "edges_before": edges_before,
        "edges_after": edges_after,
        "parameter_reduction": float(1.0 - edges_after / max(edges_before, 1)),
        **decision,
    }
    return (pruned_fit if decision["accepted"] else correction_fit), report


def fit_selective_uncertainty(corrected_oof: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    policy, metrics = fit_uncertainty(corrected_oof, config)
    policy["source_prediction_column"] = "corrected_predicted_rul"
    policy["gate_state_recalibrated"] = True
    return policy, metrics


def fit_selective_abstention(corrected_oof: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    policy, metrics = fit_abstention(corrected_oof, config)
    policy["source_prediction_column"] = "corrected_predicted_rul"
    policy["uses_gate_features"] = True
    return policy, metrics


def fit_selective_maintenance(corrected_oof: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    policy, metrics = select_maintenance_policy(corrected_oof, config)
    policy["source_prediction_column"] = "corrected_predicted_rul"
    return policy, metrics


def apply_maintenance_with_optional_labels(frame: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    point = result["corrected_predicted_rul"].astype(float)
    action = np.select(
        [point <= float(policy["urgent_threshold"]), point <= float(policy["schedule_threshold"]), point <= float(policy["inspection_threshold"])],
        ["URGENT_ENGINEERING_REVIEW", "SCHEDULE_MAINTENANCE", "PLAN_INSPECTION"],
        default="CONTINUE_MONITORING",
    )
    result["maintenance_action"] = action
    if "abstain_flag" in result:
        result.loc[result["abstain_flag"].astype(bool), "maintenance_action"] = "ABSTAIN_AND_REVIEW"
    if "true_rul" in result:
        result["safety_state"] = safety_state(result["true_rul"])
    return result


def paired_engine_alignment(*frames: pd.DataFrame) -> dict[str, Any]:
    key_sets = [set(frame["subset"].astype(str) + "::" + frame["global_engine_id"].astype(str)) for frame in frames]
    common = set.intersection(*key_sets) if key_sets else set()
    return {"aligned": all(keys == common for keys in key_sets), "common_engine_count": int(len(common)), "input_engine_counts": [int(len(keys)) for keys in key_sets]}


def paired_bootstrap_selective(phase5c: pd.DataFrame, phase5d: pd.DataFrame, phase5d1: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    key = ["subset", "global_engine_id"]
    base = phase5c[key + ["true_rul", "predicted_rul"]].rename(columns={"predicted_rul": "phase5c_pred"})
    d = phase5d[key + ["corrected_predicted_rul"]].rename(columns={"corrected_predicted_rul": "phase5d_pred"})
    s = phase5d1[key + ["corrected_predicted_rul", "maintenance_action"]].rename(columns={"corrected_predicted_rul": "phase5d1_pred", "maintenance_action": "phase5d1_action"})
    merged = base.merge(d, on=key, how="inner").merge(s, on=key, how="inner")
    true = merged["true_rul"].to_numpy(dtype=float)
    critical = true <= float(config["maintenance"]["critical_rul_max"])
    final_pred = merged["phase5d1_pred"].to_numpy(dtype=float)

    def nasa_contribution(pred: np.ndarray) -> np.ndarray:
        error = np.clip(pred - true, -100.0, 100.0)
        return np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)

    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    rows = []
    n = len(merged)
    sample_indices = rng.integers(0, n, size=(int(config["bootstrap"]["iterations"]), n))
    for comparator, column in [("phase5c_vs_phase5d1", "phase5c_pred"), ("phase5d_vs_phase5d1", "phase5d_pred")]:
        pred = merged[column].to_numpy(dtype=float)
        arrays = {
            "absolute_error": np.abs(pred - true) - np.abs(final_pred - true),
            "squared_error": (pred - true) ** 2 - (final_pred - true) ** 2,
            "nasa_contribution": nasa_contribution(pred) - nasa_contribution(final_pred),
            "optimistic_indicator": ((pred - true) > 0).astype(float) - ((final_pred - true) > 0).astype(float),
            "severe_optimistic_indicator": ((pred - true) > 25).astype(float) - ((final_pred - true) > 25).astype(float),
            "critical_miss_indicator": (critical & (pred > float(config["maintenance"]["critical_rul_max"]))).astype(float) - (critical & (final_pred > float(config["maintenance"]["critical_rul_max"]))).astype(float),
            "prediction_change_indicator": (np.abs(pred - final_pred) > 1e-8).astype(float),
        }
        for metric_name, deltas in arrays.items():
            observed = float(np.mean(deltas))
            samples = deltas[sample_indices].mean(axis=1)
            lo, hi = np.quantile(samples, [0.025, 0.975])
            rows.append({"comparison": comparator, "metric": metric_name, "point_difference_comparator_minus_phase5d1": observed, "ci_lower": float(lo), "ci_upper": float(hi), "probability_phase5d1_improves": float(np.mean(samples > 0.0)), "interval_excludes_zero": bool(lo > 0 or hi < 0), "engine_alignment_count": int(n)})
    return pd.DataFrame(rows)


def gate_feature_importance(fit: GateFit) -> list[dict[str, Any]]:
    names = list(fit.preprocessor["feature_names"])
    candidate = str(fit.candidate["candidate_id"])
    if candidate in {"logistic", "isotonic_logistic"}:
        model = fit.model.logistic if candidate == "isotonic_logistic" else fit.model
        coef = model.coef_[0]
        order = np.argsort(np.abs(coef))[::-1][:10]
        return [{"feature_name": names[int(index)], "importance": float(coef[int(index)])} for index in order]
    if candidate == "shallow_tree":
        values = fit.model.feature_importances_
        order = np.argsort(values)[::-1][:10]
        return [{"feature_name": names[int(index)], "importance": float(values[int(index)])} for index in order]
    if candidate == "sparse_additive_kan":
        importance = edge_importance_frame(fit.model, names)
        top = importance.groupby("feature_name", observed=False)["edge_importance"].sum().sort_values(ascending=False).head(10)
        return [{"feature_name": str(name), "importance": float(value)} for name, value in top.items()]
    return [{"feature_name": name, "importance": float(weight)} for name, weight in getattr(fit.model, "weights", {}).items()]


def local_explanations(benchmark: pd.DataFrame, gate_fit: GateFit, correction_fit: CorrectionFit, config: dict[str, Any]) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    key_specs = {
        "previously_missed_critical_engine_corrected_successfully": benchmark[(benchmark["true_rul"] <= 15) & (benchmark["base_predicted_rul"] > 15) & (benchmark["corrected_predicted_rul"] <= 15)].index[:1],
        "previously_missed_critical_engine_still_missed": benchmark[(benchmark["true_rul"] <= 15) & (benchmark["base_predicted_rul"] > 15) & (benchmark["corrected_predicted_rul"] > 15)].index[:1],
        "critical_engine_where_gate_failed": benchmark[(benchmark["true_rul"] <= 15) & (~benchmark["gate_active"].astype(bool))].index[:1],
        "noncritical_engine_correctly_left_unchanged": benchmark[(benchmark["true_rul"] > 30) & (~benchmark["gate_active"].astype(bool))].index[:1],
        "noncritical_engine_unnecessarily_corrected": benchmark[(benchmark["true_rul"] > 30) & (benchmark["gate_active"].astype(bool))].index[:1],
        "largest_safe_downward_correction": benchmark[benchmark["true_rul"] > 60]["downward_correction"].idxmax() if (benchmark["true_rul"] > 60).any() else None,
        "correction_hitting_maximum_bound": benchmark["downward_correction"].idxmax() if len(benchmark) else None,
    }
    correction_x, _ = transform_selective_frame(benchmark, correction_fit.preprocessor, config)
    for name, idx in key_specs.items():
        if isinstance(idx, pd.Index):
            if len(idx) == 0:
                continue
            index_value = int(idx[0])
        elif idx is None or pd.isna(idx):
            continue
        else:
            index_value = int(idx)
        row = benchmark.loc[index_value]
        payload = {
            "subset": row["subset"],
            "global_engine_id": row["global_engine_id"],
            "true_rul": float(row["true_rul"]),
            "base_rul": float(row["base_predicted_rul"]),
            "gate_probability": float(row["gate_probability"]),
            "gate_threshold": float(row["gate_threshold"]),
            "activated": bool(row["gate_active"]),
            "kan_downward_correction": float(row["downward_correction"]),
            "corrected_rul": float(row["corrected_predicted_rul"]),
            "top_gate_features": gate_feature_importance(gate_fit),
            "support_category": "IN_SUPPORT" if float(row.get("domain_support_score", 1.0)) >= 0.5 else "LOW_SUPPORT",
            "interval_width": float(row.get("interval_width_90", 0.0)),
        }
        if correction_fit.candidate["candidate_type"] == "one_sided_kan":
            payload["top_kan_contribution_features"] = local_explanation(correction_fit.model, correction_x[index_value], correction_fit.preprocessor["feature_names"]).get("top_contributions", [])
        else:
            payload["top_kan_contribution_features"] = []
        examples[name] = payload
    return examples


def freeze_decision_selective(benchmark: dict[str, Any], phase5c_base: dict[str, Any], safety: dict[str, Any], source_ok: bool, config: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if safety["missed_critical_count"] >= 25:
        reasons.append("critical_misses_not_below_phase5c_25")
    if safety["missed_critical_count"] > 21:
        reasons.append("critical_misses_worse_than_phase5d_21")
    if safety.get("new_critical_misses", 0) > 0:
        reasons.append("new_critical_misses_introduced")
    if safety["operational_critical_recall"] <= 0.7941:
        reasons.append("operational_recall_not_above_phase5d")
    if benchmark["rmse"] > phase5c_base["rmse"] + float(config["freeze"]["maximum_rmse_increase"]):
        reasons.append("rmse_noninferiority_failed")
    if benchmark["mae"] > phase5c_base["mae"] + float(config["freeze"]["maximum_mae_increase"]):
        reasons.append("mae_noninferiority_failed")
    if benchmark["severe_optimistic_rate"] > phase5c_base["severe_optimistic_rate"] + float(config["selection"]["severe_optimistic_tolerance"]):
        reasons.append("severe_optimistic_rate_worse_than_phase5c")
    if benchmark["gate_activation_rate"] > float(config["freeze"]["maximum_gate_activation"]):
        reasons.append("gate_activation_above_limit")
    if safety["total_review_workload"] > float(config["freeze"]["maximum_total_review_rate"]):
        reasons.append("review_workload_above_limit")
    if not source_ok:
        reasons.append("source_hashes_changed")
    return {
        "freeze_decision": "READY_TO_FREEZE" if not reasons else "NOT_READY",
        "reasons": reasons,
        "recommendation": "Freeze only if the selective safety intervention improves critical misses without Phase 5C accuracy regression." if reasons else "Selective AeroKAN is ready to freeze.",
    }


def make_figures(
    reports: Path,
    gate_metrics_frame: pd.DataFrame,
    corrected_oof: pd.DataFrame,
    benchmark: pd.DataFrame,
    edge_importance: pd.DataFrame,
    curves: pd.DataFrame,
    pruning: pd.DataFrame,
    bootstrap: pd.DataFrame,
    maintenance_candidates: pd.DataFrame,
    summary: dict[str, Any],
) -> list[str]:
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(name: str) -> None:
        path = fig_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        paths.append(str(path))

    y = corrected_oof["dangerous_optimism"].to_numpy(dtype=int) if "dangerous_optimism" in corrected_oof else dangerous_optimism_target(corrected_oof, {"dangerous_optimism": {"true_rul_max": 30, "optimism_threshold": 10.0}})
    p = corrected_oof["gate_probability"].to_numpy(dtype=float)
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        plt.figure(figsize=(6, 4)); plt.plot(fpr, tpr); plt.xlabel("False positive rate"); plt.ylabel("Recall"); save("gate_roc_curve.png")
        plt.figure(figsize=(6, 4)); plt.plot(recall, precision); plt.xlabel("Recall"); plt.ylabel("Precision"); save("gate_precision_recall_curve.png")
    else:
        plt.figure(figsize=(6, 4)); plt.text(0.1, 0.5, "Single gate class in OOF data"); plt.axis("off"); save("gate_roc_curve.png")
        plt.figure(figsize=(6, 4)); plt.text(0.1, 0.5, "Single gate class in OOF data"); plt.axis("off"); save("gate_precision_recall_curve.png")
    bins = pd.cut(corrected_oof["gate_probability"], bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = corrected_oof.assign(_bin=bins).groupby("_bin", observed=False).agg(probability=("gate_probability", "mean"), event=("dangerous_optimism", "mean")).dropna()
    plt.figure(figsize=(6, 4)); plt.plot(calibration["probability"], calibration["event"], marker="o"); plt.plot([0, 1], [0, 1], linestyle="--"); save("gate_calibration_curve.png")
    rows = []
    for threshold in np.linspace(0, 1, 50):
        active = p >= threshold
        rows.append({"activation": float(active.mean()), "recall": float((active & (y == 1)).sum() / max((y == 1).sum(), 1))})
    recall_frame = pd.DataFrame(rows).sort_values("activation")
    plt.figure(figsize=(6, 4)); plt.plot(recall_frame["activation"], recall_frame["recall"]); save("dangerous_recall_vs_activation_rate.png")
    corrected_oof["rul_band_plot"] = pd.cut(corrected_oof["true_rul"], bins=[-np.inf, 15, 30, 60, 90, np.inf], labels=["0_15", "16_30", "31_60", "61_90", "90_plus"])
    plt.figure(figsize=(7, 4)); corrected_oof.groupby("rul_band_plot", observed=False)["gate_active"].mean().plot(kind="bar"); save("gate_activation_by_rul_band.png")
    plt.figure(figsize=(6, 4)); plt.scatter(corrected_oof["gate_probability"], corrected_oof["downward_correction"], s=6, alpha=0.35); save("correction_magnitude_by_gate_probability.png")
    plt.figure(figsize=(7, 4)); corrected_oof.boxplot(column="downward_correction", by="rul_band_plot", rot=30); plt.suptitle(""); save("correction_magnitude_by_true_rul_band.png")
    plt.figure(figsize=(6, 4)); corrected_oof["gate_active"].value_counts().sort_index().plot(kind="bar"); save("unchanged_vs_corrected_engine_distribution.png")
    plt.figure(figsize=(6, 4)); corrected_oof[corrected_oof["true_rul"] <= 15]["downward_correction"].plot(kind="hist", bins=30); save("critical_correction_distribution.png")
    plt.figure(figsize=(6, 4)); corrected_oof[corrected_oof["true_rul"] > 60]["downward_correction"].plot(kind="hist", bins=30); save("safe_row_unnecessary_correction_distribution.png")
    if not edge_importance.empty:
        top = edge_importance.groupby("feature_name", observed=False)["edge_importance"].sum().sort_values(ascending=False).head(20)
        plt.figure(figsize=(8, 5)); top.iloc[::-1].plot(kind="barh"); save("additive_kan_feature_importance.png")
    else:
        plt.figure(figsize=(6, 4)); plt.text(0.1, 0.5, "No active KAN selected"); plt.axis("off"); save("additive_kan_feature_importance.png")
    if not curves.empty:
        plt.figure(figsize=(8, 5))
        for name, group in curves.groupby("feature_name", observed=False):
            plt.plot(group["normalized_value"], group["contribution"], label=str(name)[:20])
        plt.legend(fontsize=6); save("learned_kan_edge_curves.png")
        plt.figure(figsize=(7, 4)); curves.groupby("feature_name", observed=False)["contribution"].std().sort_values(ascending=False).head(20).plot(kind="bar"); save("kan_curve_stability.png")
    else:
        plt.figure(figsize=(6, 4)); plt.text(0.1, 0.5, "No KAN curves"); plt.axis("off"); save("learned_kan_edge_curves.png")
        plt.figure(figsize=(6, 4)); plt.text(0.1, 0.5, "No KAN curves"); plt.axis("off"); save("kan_curve_stability.png")
    plt.figure(figsize=(6, 4)); pruning.set_index("candidate_id", drop=False)[["edges_before", "edges_after"]].plot(kind="bar", ax=plt.gca()) if not pruning.empty and "edges_before" in pruning else plt.text(0.1, 0.5, "No pruning"); save("pruning_fidelity.png")
    plt.figure(figsize=(6, 4)); plt.scatter(benchmark["base_predicted_rul"], benchmark["corrected_predicted_rul"], s=12, alpha=0.5); plt.xlabel("Base"); plt.ylabel("Selective"); save("base_vs_selective_corrected_rul.png")
    if "corrected_residual" in benchmark:
        residual_frame = pd.DataFrame(
            {
                "phase5c_residual": benchmark["base_predicted_rul"].astype(float) - benchmark["true_rul"].astype(float),
                "phase5d1_residual": benchmark["corrected_residual"].astype(float),
            }
        )
        plt.figure(figsize=(7, 4)); residual_frame.plot(kind="hist", bins=30, alpha=0.5, ax=plt.gca()); save("phase5c_phase5d_phase5d1_residuals.png")
    missed = benchmark[(benchmark["true_rul"] <= 15) & (benchmark["base_predicted_rul"] > 15)]
    plt.figure(figsize=(7, 4)); missed[["base_predicted_rul", "corrected_predicted_rul"]].head(30).plot(kind="bar", ax=plt.gca()); save("previously_missed_critical_engine_corrections.png")
    failed = benchmark[(benchmark["true_rul"] <= 15) & (~benchmark["gate_active"].astype(bool))]
    plt.figure(figsize=(7, 4)); failed[["base_predicted_rul", "corrected_predicted_rul"]].head(30).plot(kind="bar", ax=plt.gca()); save("gate_failed_critical_engines.png")
    coverage = [summary["uncertainty_metrics"].get(f"coverage_{level}", 0.0) for level in [80, 90, 95]]
    plt.figure(figsize=(6, 4)); plt.plot([0.8, 0.9, 0.95], coverage, marker="o"); plt.plot([0.8, 0.9, 0.95], [0.8, 0.9, 0.95], linestyle="--"); save("coverage_vs_nominal_level.png")
    plt.figure(figsize=(6, 4)); benchmark.groupby("gate_active", observed=False)["covered_90"].mean().plot(kind="bar"); save("coverage_by_gate_state.png")
    plt.figure(figsize=(7, 4)); corrected_oof.sort_values("gate_probability")["corrected_absolute_error"].rolling(200, min_periods=20).mean().plot(); save("risk_coverage_curve.png")
    plt.figure(figsize=(7, 4)); maintenance_candidates.plot.scatter(x="total_review_workload", y="operational_critical_recall", ax=plt.gca()) if not maintenance_candidates.empty else plt.text(0.1, 0.5, "No maintenance candidates"); save("maintenance_recall_vs_workload.png")
    if not bootstrap.empty:
        subset = bootstrap.head(12)
        y_pos = np.arange(len(subset))
        plt.figure(figsize=(8, 5)); plt.errorbar(subset["point_difference_comparator_minus_phase5d1"], y_pos, xerr=[subset["point_difference_comparator_minus_phase5d1"] - subset["ci_lower"], subset["ci_upper"] - subset["point_difference_comparator_minus_phase5d1"]], fmt="o"); plt.yticks(y_pos, subset["comparison"] + " " + subset["metric"]); save("paired_bootstrap_forest_plot.png")
    else:
        plt.figure(figsize=(6, 4)); plt.text(0.1, 0.5, "No bootstrap rows"); plt.axis("off"); save("paired_bootstrap_forest_plot.png")
    plt.figure(figsize=(7, 3)); plt.axis("off"); plt.text(0.02, 0.55, f"Freeze: {summary['freeze_decision']['freeze_decision']}\nMissed critical: {summary['benchmark_safety_metrics']['missed_critical_count']}\nActivation: {summary['benchmark_metrics']['gate_activation_rate']:.3f}", fontsize=12); save("freeze_readiness_summary.png")
    return paths


def write_note(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 5D.1 Selective AeroKAN Safety Results",
        "",
        f"Freeze decision: `{summary['freeze_decision']['freeze_decision']}`",
        f"Locked gate: `{summary['locked_gate']['candidate_id']}` at threshold `{summary['locked_gate']['threshold']:.4f}`",
        f"Locked correction: `{summary['locked_correction']['candidate_id']}`",
        f"Benchmark RMSE: `{summary['benchmark_metrics']['rmse']:.4f}`",
        f"Benchmark missed critical count: `{summary['benchmark_safety_metrics']['missed_critical_count']}`",
        f"Benchmark unchanged rate: `{summary['benchmark_metrics']['unchanged_rate']:.4f}`",
        "",
        "The Phase 5C Transformer was not retrained. Benchmark labels were joined only after the pre-benchmark lock manifest was written.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_prebenchmark_lock_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    manifest = dict(payload)
    manifest["benchmark_labels_accessed_before_lock"] = False
    manifest["lock_timestamp"] = pd.Timestamp.utcnow().isoformat()
    manifest["lock_hash"] = stable_hash({key: value for key, value in manifest.items() if key != "lock_hash"})
    atomic_write_json(path, manifest)
    manifest["written_path"] = str(path)
    manifest["written_sha256"] = file_sha256(path)
    return manifest


def failure_summary(exc: BaseException) -> dict[str, Any]:
    return {"status": "failed", "exception_type": type(exc).__name__, "message": str(exc), "benchmark_labels_excluded_from_selection": True}


def run_validate_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root()
    dirs = resolve_dirs(config, root)
    manifest = build_source_manifest(config, root)
    validation = validate_sources(config, root, manifest)
    return {
        "status": validation["status"],
        "source_dirs_exist": all(path.exists() for key, path in dirs.items() if key.endswith("_reports") or key.endswith("_artifacts")),
        "output_reports_dir": str(dirs["reports"]),
        "output_artifacts_dir": str(dirs["artifacts"]),
        "backbone_frozen": bool(config["backbone"]["frozen"]),
        "missing_required_artifacts": validation["missing_required_artifacts"],
        "benchmark_labels_excluded_from_selection": True,
    }


def run_dry_run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    candidates = build_correction_candidate_registry(config)
    manifest = build_source_manifest(config, project_root())
    return {
        "status": "dry_run_complete",
        "gate_candidates": ["no_gate"] + [candidate for candidate in config["gate"]["candidates"] if candidate != "no_gate"],
        "correction_candidate_count": len(candidates),
        "correction_candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "required_source_artifact_count": len([row for row in manifest if row["required"]]),
        "missing_required_artifacts": [row["artifact_key"] for row in manifest if row["required"] and not row["exists"]],
        "prebenchmark_lock_required": True,
        "benchmark_labels_excluded_from_selection": True,
    }


def run_smoke_test(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    predictions, sensors = aerokan_synthetic_frame(n_engines=10, windows=4)
    features = build_named_features(predictions, sensors, config)
    features = add_dangerous_targets(features, config, max(float(value) for value in config["correction"]["bounds"]))
    split_train, split_val, split = split_by_engine(features, 0.6, 77)
    gate_fit = fit_gate_candidate("logistic", split_train, config)
    split_train_gate = add_gate_columns(split_train, gate_fit, config)
    correction_candidates = [candidate for candidate in build_correction_candidate_registry(config, smoke=True) if candidate["candidate_type"] in {"constant", "linear_nonnegative", "mlp_magnitude", "one_sided_kan"}][:4]
    fits = [fit_correction_candidate(candidate, split_train_gate, config, 1) for candidate in correction_candidates]
    corrected_examples = []
    for fit in fits:
        corrected = selective_corrected_predictions(split_val, gate_fit, fit, config)
        corrected_examples.append(verify_one_sided_property(corrected))
    kan_gate = fit_gate_candidate("sparse_additive_kan", split_train, config)
    kan_candidate = next(candidate for candidate in build_correction_candidate_registry(config, smoke=True) if candidate["candidate_type"] == "one_sided_kan")
    kan_fit = fit_correction_candidate(kan_candidate, add_gate_columns(split_train, kan_gate, config), config, 1)
    kan_corrected = selective_corrected_predictions(split_val, kan_gate, kan_fit, config)
    pruned_fit, pruning = prune_if_accepted(kan_fit, add_gate_columns(split_val, kan_gate, config), kan_gate, config)
    uncertainty_policy, _ = fit_selective_uncertainty(kan_corrected.assign(corrected_residual=kan_corrected["corrected_predicted_rul"] - kan_corrected["true_rul"]), config)
    with_uncertainty = apply_uncertainty(kan_corrected, uncertainty_policy, config)
    abstention_policy, _ = fit_selective_abstention(with_uncertainty, config)
    with_abstention = apply_abstention(with_uncertainty, abstention_policy)
    maintenance_policy, _ = fit_selective_maintenance(with_abstention, config)
    with tempfile.TemporaryDirectory() as temp_dir:
        lock = write_prebenchmark_lock_manifest(Path(temp_dir) / "prebenchmark_lock_manifest.json", {"gate_model_family": "logistic", "correction_model": kan_candidate, "benchmark_labels_accessed_before_lock": False})
    return {
        "status": "smoke_complete",
        "synthetic_only": True,
        "engine_overlap_count": split["engine_overlap_count"],
        "gate_candidate": gate_fit.candidate["candidate_id"],
        "sparse_kan_gate_probability_finite": bool(np.isfinite(predict_gate_probability(kan_gate, split_val, config)[0]).all()),
        "correction_candidates_exercised": [fit.candidate["candidate_id"] for fit in fits],
        "one_sided_checks": corrected_examples,
        "kan_one_sided_check": verify_one_sided_property(kan_corrected),
        "pruning_accepted": bool(pruning["accepted"]),
        "unpruned_model_remains_valid": isinstance(pruned_fit.model, nn.Module),
        "uncertainty_method": uncertainty_policy["method_id"],
        "abstention_method": abstention_policy["method_id"],
        "maintenance_policy": maintenance_policy["policy_id"],
        "prebenchmark_lock_written": bool(lock["written_sha256"]),
        "benchmark_leakage": False,
        "backbone_training_called": TRANSFORMER_TRAINING_CALLED,
    }


def run_full_run(config_path: str | Path) -> dict[str, Any]:
    start = time.perf_counter()
    config = load_config(config_path)
    root = project_root()
    reports, artifacts = prepare_outputs(config, root)
    dirs = resolve_dirs(config, root)

    manifest_before = build_source_manifest(config, root)
    atomic_write_json(reports / "source_artifact_manifest.json", {"artifacts": manifest_before, "phase": "before"})
    source_validation = validate_sources(config, root, manifest_before)
    atomic_write_json(reports / "source_validation.json", source_validation)

    phase5c_cv = pd.read_csv(dirs["phase5c_reports"] / "cv_predictions.csv")
    phase5c_benchmark_features = read_phase5c_benchmark_features(dirs["phase5c_reports"] / "benchmark_predictions.csv")
    train_sensors = load_training_sensor_frame(config, root)
    benchmark_sensors = load_benchmark_sensor_frame(config, root)

    oof_features = build_named_features(phase5c_cv, train_sensors, config)
    definition = dangerous_definition_report(oof_features, config)
    atomic_write_json(reports / "dangerous_optimism_definition.json", definition)
    oof_features = add_dangerous_targets(oof_features, config, max(float(value) for value in config["correction"]["bounds"]))
    selection_features = cap_windows_per_engine(oof_features, config, int(config["selection"]["screening_seed"]))
    benchmark_features = build_named_features(phase5c_benchmark_features, benchmark_sensors, config)

    gate_features = gate_candidate_feature_names(config)
    correction_features = correction_candidate_feature_names(config)
    gate_registry = {
        "gate_candidate_features": gate_features,
        "final_gate_feature_limit": int(config["gate"]["maximum_features"]),
        "feature_leakage_audit": audit_feature_leakage(gate_features),
        "selected_using_benchmark": False,
    }
    atomic_write_json(reports / "gate_feature_registry.json", gate_registry)
    atomic_write_json(reports / "feature_leakage_audit.json", {"gate": audit_feature_leakage(gate_features), "correction": audit_feature_leakage(correction_features), "benchmark_labels_used_before_lock": False})

    gate_screening, gate_calibration, locked_gate_screen, gate_split = screen_risk_gate_candidates(selection_features, config)
    gate_screening.to_csv(reports / "gate_candidate_metrics.csv", index=False)
    gate_calibration.to_csv(reports / "gate_calibration_metrics.csv", index=False)

    correction_candidates = build_correction_candidate_registry(config)
    atomic_write_json(reports / "correction_candidate_registry.json", {"candidates": correction_candidates, "selected_using_benchmark": False})
    correction_screening, _, correction_split = screen_correction_candidates(selection_features, locked_gate_screen, correction_candidates, config)
    correction_screening.to_csv(reports / "correction_screening_metrics.csv", index=False)
    finalist_ids = select_correction_finalists(correction_screening, config)
    finalist_cv = run_finalist_cross_validation(selection_features, str(locked_gate_screen.candidate["candidate_id"]), correction_candidates, finalist_ids, config)
    finalist_cv.to_csv(reports / "finalist_cross_validation_metrics.csv", index=False)
    locked_correction_selection = choose_locked_correction(finalist_cv, config)
    selected_candidate = {candidate["candidate_id"]: candidate for candidate in correction_candidates}[str(locked_correction_selection["candidate_id"])]

    final_gate = fit_gate_candidate(str(locked_gate_screen.candidate["candidate_id"]), selection_features, config)
    final_train_with_gate = add_gate_columns(selection_features, final_gate, config)
    final_correction = fit_correction_candidate(selected_candidate, final_train_with_gate, config, int(config["training"]["final_epochs"]))
    final_correction, pruning_report = prune_if_accepted(final_correction, final_train_with_gate, final_gate, config)
    pruning_frame = pd.DataFrame([{**pruning_report, "candidate_id": selected_candidate["candidate_id"]}])
    pruning_frame.to_csv(reports / "kan_pruning_results.csv", index=False)

    corrected_oof = selective_corrected_predictions(oof_features, final_gate, final_correction, config)
    corrected_oof["fold_marker"] = corrected_oof.get("fold", pd.Series(0, index=corrected_oof.index)).astype(str)
    corrected_oof["seed_marker"] = corrected_oof.get("seed", pd.Series(0, index=corrected_oof.index)).astype(str)
    corrected_oof.to_csv(reports / "corrected_oof_predictions.csv", index=False)
    selective_oof_metrics = selective_metrics(corrected_oof, config)
    pd.DataFrame([selective_oof_metrics]).to_csv(reports / "selective_correction_metrics.csv", index=False)

    uncertainty_policy, uncertainty_metrics = fit_selective_uncertainty(corrected_oof, config)
    atomic_write_json(reports / "locked_uncertainty_method.json", uncertainty_policy)
    atomic_write_json(reports / "uncertainty_metrics.json", uncertainty_metrics)
    corrected_oof = apply_uncertainty(corrected_oof, uncertainty_policy, config)
    abstention_policy, abstention_metrics = fit_selective_abstention(corrected_oof, config)
    atomic_write_json(reports / "locked_abstention_policy.json", abstention_policy)
    atomic_write_json(reports / "abstention_metrics.json", abstention_metrics)
    corrected_oof = apply_abstention(corrected_oof, abstention_policy)
    maintenance_policy, maintenance_candidates = fit_selective_maintenance(corrected_oof, config)
    maintenance_candidates.to_csv(reports / "maintenance_policy_candidates.csv", index=False)
    atomic_write_json(reports / "locked_maintenance_policy.json", maintenance_policy)
    corrected_oof = apply_maintenance_with_optional_labels(corrected_oof, maintenance_policy)

    if final_correction.candidate["candidate_type"] == "one_sided_kan":
        layers = collect_kan_layers(final_correction.model)
        curves = univariate_curve_frame(layers[0], final_correction.preprocessor["feature_names"]) if layers else pd.DataFrame(columns=["feature_name", "normalized_value", "contribution"])
        edge_importance = edge_importance_frame(final_correction.model, final_correction.preprocessor["feature_names"])
    else:
        curves = pd.DataFrame(columns=["feature_name", "normalized_value", "contribution"])
        edge_importance = pd.DataFrame(columns=["feature_name", "edge_importance", "active"])
    curves.to_csv(reports / "kan_curve_stability.csv", index=False)
    edge_importance.to_csv(reports / "kan_edge_importance.csv", index=False)
    symbolic = approximate_curves(curves, fidelity_rmse=0.05) if not curves.empty else pd.DataFrame(columns=["feature_name", "function", "coefficients", "approximation_rmse", "maximum_deviation", "accepted"])
    symbolic.to_csv(reports / "kan_symbolic_approximations.csv", index=False)

    gate_model_path = artifacts / "gate_model.pkl"
    with gate_model_path.open("wb") as handle:
        pickle.dump({"candidate": final_gate.candidate, "model": final_gate.model, "threshold": final_gate.threshold}, handle)
    gate_preprocessor_path = artifacts / "gate_preprocessor.pkl"
    with gate_preprocessor_path.open("wb") as handle:
        pickle.dump(final_gate.preprocessor, handle)
    correction_preprocessor_path = artifacts / "kan_feature_preprocessor.pkl"
    with correction_preprocessor_path.open("wb") as handle:
        pickle.dump(final_correction.preprocessor, handle)
    correction_checkpoint_path = artifacts / "one_sided_kan_checkpoint.pt"
    if isinstance(final_correction.model, nn.Module):
        torch.save({"candidate": final_correction.candidate, "state_dict": final_correction.model.state_dict(), "feature_names": final_correction.preprocessor["feature_names"]}, correction_checkpoint_path)
    else:
        with correction_checkpoint_path.open("wb") as handle:
            pickle.dump({"candidate": final_correction.candidate, "model": final_correction.model, "feature_names": final_correction.preprocessor["feature_names"]}, handle)
    atomic_write_json(artifacts / "healthy_baseline_metadata.json", {"gate": final_gate.preprocessor["healthy_baselines"], "correction": final_correction.preprocessor["healthy_baselines"], "definition": final_correction.preprocessor["healthy_row_definition"]})
    atomic_write_json(artifacts / "pruning_mask.json", pruning_report)
    with (artifacts / "uncertainty_model.pkl").open("wb") as handle:
        pickle.dump(uncertainty_policy, handle)
    with (artifacts / "abstention_model.pkl").open("wb") as handle:
        pickle.dump(abstention_policy, handle)
    with (artifacts / "maintenance_policy.pkl").open("wb") as handle:
        pickle.dump(maintenance_policy, handle)

    source_model_hash = next(row["sha256"] for row in manifest_before if row["artifact_key"] == "phase5c_checkpoint")
    lock_payload = {
        "frozen_phase5c_checkpoint_hash": source_model_hash,
        "gate_model_family": final_gate.candidate["candidate_id"],
        "gate_features": final_gate.preprocessor["feature_names"],
        "gate_threshold": float(final_gate.threshold),
        "gate_coefficients_or_rules": gate_feature_importance(final_gate),
        "dangerous_event_definition": definition,
        "kan_architecture": final_correction.candidate,
        "kan_feature_names": final_correction.preprocessor["feature_names"],
        "spline_grid": final_correction.candidate.get("grid_size"),
        "spline_degree": final_correction.candidate.get("spline_degree"),
        "correction_bound": final_correction.candidate.get("correction_bound"),
        "loss_weights": {
            "zero": final_correction.candidate.get("zero_weight"),
            "critical": final_correction.candidate.get("critical_weight"),
            "sparsity": final_correction.candidate.get("sparsity"),
            "smoothness": final_correction.candidate.get("smoothness"),
        },
        "active_edges": int(edge_importance[edge_importance.get("active", False)]["active"].sum()) if not edge_importance.empty and "active" in edge_importance else 0,
        "pruning_status": pruning_report,
        "uncertainty_method": uncertainty_policy,
        "abstention_policy": abstention_policy,
        "maintenance_policy": maintenance_policy,
        "cross_validation_results": locked_correction_selection,
        "selection_criteria": config["selection"],
        "seeds": config["selection"]["seeds"],
        "gate_model_hash": file_sha256(gate_model_path),
        "correction_checkpoint_hash": file_sha256(correction_checkpoint_path),
        "benchmark_labels_accessed_before_lock": False,
    }
    lock_manifest = write_prebenchmark_lock_manifest(reports / "prebenchmark_lock_manifest.json", lock_payload)

    benchmark_labels = read_phase5c_benchmark_labels(dirs["phase5c_reports"] / "benchmark_predictions.csv")
    benchmark_features_with_labels = benchmark_features.merge(benchmark_labels, on=["subset", "global_engine_id"], how="left", validate="one_to_one")
    benchmark_corrected = selective_corrected_predictions(benchmark_features_with_labels, final_gate, final_correction, config)
    benchmark_corrected = apply_uncertainty(benchmark_corrected, uncertainty_policy, config)
    benchmark_corrected = apply_abstention(benchmark_corrected, abstention_policy)
    benchmark_corrected = apply_maintenance_with_optional_labels(benchmark_corrected, maintenance_policy)
    benchmark_corrected.to_csv(reports / "benchmark_predictions.csv", index=False)

    benchmark_metrics = selective_metrics(benchmark_corrected, config)
    benchmark_by_subset = benchmark_point_by_subset(benchmark_corrected)
    benchmark_by_subset.to_csv(reports / "benchmark_metrics_by_subset.csv", index=False)
    atomic_write_json(reports / "benchmark_metrics.json", benchmark_metrics)
    benchmark_safety = maintenance_metrics(benchmark_corrected)
    phase5c_benchmark_full = pd.read_csv(dirs["phase5c_reports"] / "benchmark_predictions.csv")
    phase5d_benchmark = pd.read_csv(dirs["phase5d_reports"] / "benchmark_predictions.csv")
    previous_missed = phase5c_benchmark_full[(phase5c_benchmark_full["true_rul"] <= 15) & (phase5c_benchmark_full["predicted_rul"] > 15)][["subset", "global_engine_id"]]
    previous_missed_keys = set(previous_missed["subset"].astype(str) + "::" + previous_missed["global_engine_id"].astype(str))
    current_missed = benchmark_corrected[(benchmark_corrected["true_rul"] <= 15) & ~benchmark_corrected["maintenance_action"].isin(["URGENT_ENGINEERING_REVIEW", "ABSTAIN_AND_REVIEW"])]
    current_missed_keys = set(current_missed["subset"].astype(str) + "::" + current_missed["global_engine_id"].astype(str))
    benchmark_safety.update(
        {
            "previous_phase5c_misses_corrected": int(len(previous_missed_keys - current_missed_keys)),
            "previous_phase5c_misses_still_missed": int(len(previous_missed_keys & current_missed_keys)),
            "new_critical_misses": int(len(current_missed_keys - previous_missed_keys)),
            "gate_failed_critical_count": int(((benchmark_corrected["true_rul"] <= 15) & (~benchmark_corrected["gate_active"].astype(bool))).sum()),
            "insufficient_correction_critical_count": int(((benchmark_corrected["true_rul"] <= 15) & (benchmark_corrected["gate_active"].astype(bool)) & (benchmark_corrected["corrected_predicted_rul"] > 15)).sum()),
            "prebenchmark_lock_sha256": lock_manifest["written_sha256"],
        }
    )
    atomic_write_json(reports / "benchmark_safety_metrics.json", benchmark_safety)

    phase5c_base_metrics = point_metrics(phase5c_benchmark_full, phase5c_benchmark_full["predicted_rul"].to_numpy(dtype=float), correction=np.zeros(len(phase5c_benchmark_full)))
    phase5d_metrics = point_metrics(phase5d_benchmark, phase5d_benchmark["corrected_predicted_rul"].to_numpy(dtype=float), correction=phase5d_benchmark["kan_correction"].to_numpy(dtype=float))
    comparison = pd.DataFrame(
        [
            {"phase": "phase5c", **phase5c_base_metrics},
            {"phase": "phase5d_global_aerokan", **phase5d_metrics},
            {"phase": "phase5d1_selective_aerokan", **benchmark_metrics},
        ]
    )
    comparison.to_csv(reports / "phase5c_phase5d_phase5d1_comparison.csv", index=False)
    bootstrap = paired_bootstrap_selective(phase5c_benchmark_full, phase5d_benchmark, benchmark_corrected, config)
    bootstrap.to_csv(reports / "paired_bootstrap_results.csv", index=False)
    explanations = local_explanations(benchmark_corrected, final_gate, final_correction, config)
    atomic_write_json(reports / "local_explanations.json", explanations)

    manifest_after = build_source_manifest(config, root)
    unchanged = source_hashes_unchanged(manifest_before, manifest_after)
    freeze = freeze_decision_selective(benchmark_metrics, phase5c_base_metrics, benchmark_safety, unchanged, config)
    atomic_write_json(reports / "freeze_decision.json", freeze)

    summary = {
        "status": "completed",
        "runtime_seconds": time.perf_counter() - start,
        "source_validation": source_validation,
        "source_hashes_unchanged": unchanged,
        "backbone_frozen_verification": {
            "checkpoint_hash_before": source_model_hash,
            "checkpoint_hash_after": next(row["sha256"] for row in manifest_after if row["artifact_key"] == "phase5c_checkpoint"),
            "hash_unchanged": unchanged,
            "transformer_training_called": TRANSFORMER_TRAINING_CALLED,
        },
        "dangerous_event_definition": definition,
        "gate_screening_split": gate_split,
        "correction_screening_split": correction_split,
        "gate_candidates": gate_screening["candidate_id"].tolist(),
        "locked_gate": {"candidate_id": final_gate.candidate["candidate_id"], "threshold": float(final_gate.threshold), "features": final_gate.preprocessor["feature_names"], "metrics": final_gate.metrics},
        "correction_candidates": [candidate["candidate_id"] for candidate in correction_candidates],
        "finalists": finalist_ids,
        "locked_correction": {
            "candidate_id": final_correction.candidate["candidate_id"],
            "candidate_type": final_correction.candidate["candidate_type"],
            "correction_bound": float(final_correction.candidate.get("correction_bound", 0.0)),
            "feature_names": final_correction.preprocessor["feature_names"],
            "selection": locked_correction_selection,
        },
        "kan_interpretability": {
            "active_features": int(edge_importance[edge_importance.get("active", False)]["feature_name"].nunique()) if not edge_importance.empty and "active" in edge_importance else 0,
            "active_edges": int(edge_importance[edge_importance.get("active", False)]["active"].sum()) if not edge_importance.empty and "active" in edge_importance else 0,
            "symbolic_accepted": int(symbolic.get("accepted", pd.Series(dtype=bool)).sum()) if not symbolic.empty else 0,
        },
        "pruning": pruning_report,
        "validation_metrics": selective_oof_metrics,
        "uncertainty_metrics": uncertainty_metrics,
        "abstention_metrics": abstention_metrics,
        "maintenance_validation_metrics": maintenance_metrics(corrected_oof),
        "benchmark_metrics": benchmark_metrics,
        "benchmark_safety_metrics": benchmark_safety,
        "phase5c_base_metrics": phase5c_base_metrics,
        "phase5d_global_metrics": phase5d_metrics,
        "paired_engine_alignment": paired_engine_alignment(phase5c_benchmark_full, phase5d_benchmark, benchmark_corrected),
        "benchmark_labels_excluded_from_selection": True,
        "model_locked_before_benchmark": True,
        "prebenchmark_lock_manifest": lock_manifest,
        "environment_changed": False,
        "packages_installed": False,
        "git_used": False,
        "freeze_decision": freeze,
    }
    figures = make_figures(reports, gate_screening, corrected_oof, benchmark_corrected, edge_importance, curves, pruning_frame, bootstrap, maintenance_candidates, summary)
    summary["figures"] = figures
    summary["generated_reports"] = [str(path) for path in reports.glob("*") if path.is_file()]
    summary["generated_artifacts"] = [str(path) for path in artifacts.glob("*") if path.is_file()]
    atomic_write_json(reports / "run_summary.json", summary)
    atomic_write_json(reports / "source_artifact_manifest.json", {"artifacts": manifest_after, "verified_unchanged": unchanged})
    write_note(root / "notes" / "selective_aerokan_safety_results.md", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5D.1 selective one-sided AeroKAN safety corrector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--full-run", action="store_true")
    args = parser.parse_args(argv)
    modes = [args.validate_config, args.dry_run, args.smoke_test, args.full_run]
    if sum(bool(mode) for mode in modes) != 1:
        parser.error("Select exactly one mode.")
    try:
        if args.validate_config:
            result = run_validate_config(args.config)
        elif args.dry_run:
            result = run_dry_run(args.config)
        elif args.smoke_test:
            result = run_smoke_test(args.config)
        else:
            result = run_full_run(args.config)
    except Exception as exc:
        result = failure_summary(exc)
        print(json.dumps(json_ready(result), indent=2, sort_keys=True))
        raise
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
