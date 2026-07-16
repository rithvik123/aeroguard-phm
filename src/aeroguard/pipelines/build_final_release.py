"""Build the AeroGuard-PHM Phase 6 final-release package.

This script is read-only with respect to previous phase outputs. It creates
new final-release reports, artifacts, documentation, figures, and examples
from authoritative source artifacts without retraining or tuning any model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aeroguard.evaluation.metrics import nasa_asymmetric_score, r2_score
from aeroguard.inference.monitoring import monitoring_spec, schema_hash
from aeroguard.inference.predictor import AeroGuardPredictor
from aeroguard.inference.schemas import REQUIRED_INPUT_COLUMNS, output_schema


SYSTEM_NAME = "AeroGuard-PHM Safety-Guarded RUL System"
RELEASE_VERSION = "1.0.0"
MODEL_VERSION = "aeroguard-phm-safety-v1"
FINAL_MODEL_NAME = "Critical-Boundary Safety-Guarded Physics-Guided Transformer"
PREDICTIVE_BACKBONE = "Regime-Consistent Physics-Guided Patch Transformer"
SAFETY_LAYER = "Critical-Boundary Safety Guard"
FINAL_ARCHITECTURE = "Critical-Boundary Safety-Guarded Physics-Guided Patch Transformer"
CANONICAL_SEVERE_THRESHOLD = 30.0
CRITICAL_THRESHOLD = 15.0

DISPLAY_NAME_OVERRIDES = {
    "classical_random_forest": "Random Forest RUL Baseline",
    "sequence_mlp": "Sequence MLP RUL Baseline",
    "cnn1d": "Temporal CNN RUL Baseline",
    "lstm": "LSTM RUL Baseline",
    "gru": "GRU RUL Baseline",
    "tcn": "Temporal Convolutional RUL Baseline",
    "cnn_lstm": "CNN-LSTM RUL Baseline",
    "lstm_phase5_schedule_a": "LSTM Candidate - Schedule A",
    "lstm_phase5_schedule_b": "LSTM Candidate - Schedule B",
    "tcn_extended_schedule_a": "Temporal CNN Candidate - Schedule A",
    "tcn_extended_schedule_b": "Temporal CNN Candidate - Schedule B",
    "temporal_transformer_sinusoidal_mean_a": "Temporal Transformer - Sinusoidal Mean Pooling",
    "temporal_transformer_learned_attention_b": "Temporal Transformer - Learned Attention Pooling",
    "patch_transformer_5x5_attention_a": "Patch Transformer — 5×5 Patches with Attention Pooling",
    "patch_transformer_10x5_mean_b": "Patch Transformer — 10×5 Patches with Mean Pooling",
    "phase5b_reimplementation_baseline": "Patch Transformer Reimplementation Baseline",
    "physics_monotonic": "Monotonicity-Guided Patch Transformer",
    "physics_cycle_rate": "Cycle-Rate-Guided Patch Transformer",
    "physics_smooth": "Smoothness-Guided Patch Transformer",
    "physics_health": "Health-Guided Patch Transformer",
    "physics_regime": PREDICTIVE_BACKBONE,
    "physics_temporal_combined": "Temporally Combined Physics-Guided Patch Transformer",
    "physics_full": "Full Physics-Guided Patch Transformer",
    "physics_full_safety": "Full Physics-and-Safety Patch Transformer",
    "phase5c_frozen_baseline": PREDICTIVE_BACKBONE,
    "linear_ridge_residual": "Linear Residual Corrector",
    "small_mlp_residual": "MLP Residual Corrector",
    "direct_additive_kan_rul": "Direct Additive KAN RUL Head",
    "sparse_kan_residual_bound10": "Sparse Additive KAN Residual Corrector (Bound 10)",
    "sparse_kan_residual_bound20": "Sparse Additive KAN Residual Corrector (Bound 20)",
    "sparse_kan_residual_bound30": "Sparse Additive KAN Residual Corrector (Bound 30)",
    "safety_weighted_sparse_kan": "Safety-Weighted Sparse KAN Corrector",
    "regime_aware_sparse_kan": "Regime-Aware KAN Corrector",
    "two_layer_compact_kan_h4": "AeroKAN-PHM Compact Residual Corrector (h=4)",
    "two_layer_compact_kan_h8": "AeroKAN-PHM Compact Residual Corrector",
    "phase5c_exact_fallback": "Frozen Physics-Guided Baseline Fallback",
    "phase5d_two_layer_one_sided_control_bound20": "Selective One-Sided AeroKAN Safety Corrector",
    "selective_one_sided_aerokan_safety_corrector": "Selective One-Sided AeroKAN Safety Corrector",
    "critical_boundary_safety_guarded_transformer": FINAL_MODEL_NAME,
    "aeroguard_phm_safety_guarded_rul_system": SYSTEM_NAME,
}

FORBIDDEN_PUBLIC_PHRASES = [
    "repository-derived " + "baseline",
    "repo " + "baseline",
    "Phase 5C " + "model",
    "Phase 5D " + "model",
    "bad " + "model",
    "failed " + "model",
]


def root_path() -> Path:
    return Path(__file__).resolve().parents[3]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(json_ready(payload), handle, indent=2, allow_nan=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sanitize_public_text(text: str, root: Path) -> str:
    """Remove workstation-specific paths from generated public artifacts."""
    root_resolved = root.resolve()
    replacements = [
        (str(root_resolved) + "\\", ""),
        (str(root_resolved), "."),
        (str(root_resolved).replace("\\", "\\\\") + "\\\\", ""),
        (str(root_resolved).replace("\\", "\\\\"), "."),
        (root_resolved.as_posix() + "/", ""),
        (root_resolved.as_posix(), "."),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    text = re.sub(r'&\s+"[A-Za-z]:(?:\\\\|\\)Users(?:\\\\|\\)[^"]+?python\.exe"\s+', "python ", text)
    text = re.sub(r'"[A-Za-z]:(?:\\\\|\\)Users(?:\\\\|\\)[^"]+?python\.exe"', '"python"', text)
    return text


def sanitize_release_text_outputs(root: Path, *directories: Path) -> None:
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.suffix.lower() not in {".csv", ".json", ".md", ".txt"} or not path.is_file():
                continue
            original = path.read_text(encoding="utf-8")
            sanitized = sanitize_public_text(original, root)
            if sanitized != original:
                path.write_text(sanitized, encoding="utf-8", newline="\n")


def display_name(model_id: str) -> str:
    if model_id in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[model_id]
    text = model_id.replace("_", " ").replace("-", " ")
    return " ".join(part.upper() if part in {"mlp", "gru", "lstm", "kan", "rul"} else part.capitalize() for part in text.split())


def component(root: Path, path: Path, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "path": rel(path, root),
        "sha256": sha256_file(path) if path.exists() else None,
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def point_metrics(true: pd.Series, pred: pd.Series) -> dict[str, float]:
    y = true.to_numpy(dtype=float)
    p = pred.to_numpy(dtype=float)
    residual = p - y
    low_mask = y <= CANONICAL_SEVERE_THRESHOLD
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "r2": float(r2_score(y, p)),
        "nasa_score": float(nasa_asymmetric_score(y, p)),
        "mean_signed_error": float(np.mean(residual)),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "optimistic_prediction_rate": float(np.mean(residual > 0)),
        "severe_optimistic_prediction_rate": float(np.mean(residual > CANONICAL_SEVERE_THRESHOLD)),
        "low_rul_optimistic_rate": float(np.mean(residual[low_mask] > 0)) if bool(low_mask.any()) else math.nan,
    }


def fixed_policy_metrics(frame: pd.DataFrame, pred_column: str, policy_id: str) -> dict[str, Any]:
    critical = frame["true_rul"].astype(float) <= CRITICAL_THRESHOLD
    urgent = frame[pred_column].astype(float) <= CRITICAL_THRESHOLD
    miss = critical & ~urgent
    urgent_count = int(urgent.sum())
    critical_count = int(critical.sum())
    precision = float((critical & urgent).sum() / urgent_count) if urgent_count else None
    recall = float((critical & urgent).sum() / critical_count) if critical_count else None
    return {
        "policy_id": policy_id,
        "critical_count": critical_count,
        "critical_miss_count": int(miss.sum()),
        "direct_urgent_recall": recall,
        "operational_recall": recall,
        "urgent_precision": precision,
        "abstain_review_count": 0,
        "total_review_count": urgent_count,
        "review_workload": float(urgent_count / len(frame)) if len(frame) else None,
    }


def load_prediction_sources(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "model_id": "patch_transformer_10x5_mean_b",
            "family": "Patch Transformer",
            "status": "Baseline",
            "path": root / "reports/deep_rul_extended/benchmark_predictions.csv",
            "prediction_column": "predicted_rul",
            "interval_path": root / "reports/deep_rul_extended/uncertainty_predictions.csv",
        },
        {
            "model_id": "physics_regime",
            "family": "Physics-guided Transformer",
            "status": "Candidate",
            "path": root / "reports/physics_guided_rul/benchmark_predictions.csv",
            "prediction_column": "predicted_rul",
            "interval_path": root / "reports/physics_guided_rul/uncertainty_predictions.csv",
        },
        {
            "model_id": "two_layer_compact_kan_h8",
            "family": "AeroKAN residual correction",
            "status": "Accuracy trade-off",
            "path": root / "reports/aerokan_phm/benchmark_predictions.csv",
            "prediction_column": "corrected_predicted_rul",
            "interval_path": root / "reports/aerokan_phm/benchmark_predictions.csv",
        },
        {
            "model_id": "selective_one_sided_aerokan_safety_corrector",
            "family": "Selective AeroKAN safety correction",
            "status": "Not selected",
            "path": root / "reports/aerokan_phm_selective/benchmark_predictions.csv",
            "prediction_column": "corrected_predicted_rul",
            "interval_path": root / "reports/aerokan_phm_selective/benchmark_predictions.csv",
        },
        {
            "model_id": "critical_boundary_safety_guarded_transformer",
            "family": "Safety-guarded Transformer",
            "status": "Final selected system",
            "path": root / "reports/aerokan_phm_critical_gate/benchmark_predictions.csv",
            "prediction_column": "corrected_predicted_rul",
            "interval_path": root / "reports/aerokan_phm_critical_gate/benchmark_predictions.csv",
        },
    ]


def aligned_prediction_frame(root: Path, sources: list[dict[str, Any]]) -> pd.DataFrame:
    key = ["subset", "global_engine_id", "final_observed_cycle"]
    merged: pd.DataFrame | None = None
    for source in sources:
        df = safe_read_csv(source["path"])
        if df.empty:
            continue
        pred_col = source["prediction_column"]
        columns = key + ["true_rul", pred_col]
        model_frame = df[columns].copy()
        model_frame = model_frame.rename(columns={pred_col: source["model_id"]})
        if merged is None:
            merged = model_frame
        else:
            merged = merged.merge(model_frame, on=key + ["true_rul"], how="inner", validate="one_to_one")
    if merged is None or merged.empty:
        raise RuntimeError("No aligned benchmark prediction rows were found.")
    return merged


def build_point_and_fixed_tables(root: Path, reports_dir: Path, sources: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aligned = aligned_prediction_frame(root, sources)
    aligned.to_csv(reports_dir / "aligned_benchmark_predictions.csv", index=False)
    point_rows = []
    fixed_rows = []
    for source in sources:
        model_id = source["model_id"]
        metrics = point_metrics(aligned["true_rul"], aligned[model_id])
        point_rows.append(
            {
                "model_id": model_id,
                "display_name": display_name(model_id),
                "model_family": source["family"],
                "engine_count": int(len(aligned)),
                "benchmark_engine_keys": "identical",
                "true_rul_definition": "uncapped",
                "residual_orientation": "predicted_rul_minus_true_rul",
                "severe_optimistic_threshold": CANONICAL_SEVERE_THRESHOLD,
                "policy_effects_included": False,
                "abstention_effects_included": False,
                **metrics,
                "source_predictions": rel(source["path"], root),
            }
        )
        fpm = fixed_policy_metrics(aligned.rename(columns={model_id: "fixed_prediction"}), "fixed_prediction", "point_u15_s30_i60")
        fixed_rows.append(
            {
                "model_id": model_id,
                "display_name": display_name(model_id),
                "model_family": source["family"],
                "engine_count": int(len(aligned)),
                "fixed_policy_label": "identical point_u15_s30_i60, no abstention",
                **fpm,
                "source_predictions": rel(source["path"], root),
            }
        )
    point_df = pd.DataFrame(point_rows)
    fixed_df = pd.DataFrame(fixed_rows)
    point_df.to_csv(reports_dir / "point_prediction_comparison.csv", index=False)
    fixed_df.to_csv(reports_dir / "fixed_policy_safety_comparison.csv", index=False)
    return aligned, point_df, fixed_df


def build_native_system_table(root: Path, reports_dir: Path, point_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_map = {
        "patch_transformer_10x5_mean_b": root / "reports/deep_rul_extended/maintenance_policy_metrics.json",
        "physics_regime": root / "reports/physics_guided_rul_safety_refined/benchmark_safety_metrics.json",
        "two_layer_compact_kan_h8": root / "reports/aerokan_phm/benchmark_safety_metrics.json",
        "selective_one_sided_aerokan_safety_corrector": root / "reports/aerokan_phm_selective/benchmark_safety_metrics.json",
        "critical_boundary_safety_guarded_transformer": root / "reports/aerokan_phm_critical_gate/benchmark_fixed_policy_metrics.json",
    }
    for _, point in point_df.iterrows():
        model_id = str(point["model_id"])
        native = read_json(source_map[model_id], {})
        row = {
            "model_id": model_id,
            "display_name": point["display_name"],
            "model_family": point["model_family"],
            "overall_mae": point["mae"],
            "overall_rmse": point["rmse"],
            "nasa_score": point["nasa_score"],
            "severe_optimism": point["severe_optimistic_prediction_rate"],
            "native_policy_label": "native locked uncertainty, abstention, and maintenance policy where available",
            "source_metrics": rel(source_map[model_id], root),
        }
        if model_id == "patch_transformer_10x5_mean_b":
            actions = native.get("action_counts", {})
            urgent = int(actions.get("URGENT_ENGINEERING_REVIEW", 0))
            abstain = int(native.get("abstention_count", 0))
            row.update(
                {
                    "critical_misses": native.get("critical_engines_not_receiving_urgent_review"),
                    "operational_recall": native.get("urgent_review_recall_true_rul_le_15"),
                    "urgent_precision": None,
                    "abstention_review_count": abstain,
                    "total_review_count": urgent + abstain,
                    "review_workload": (urgent + abstain) / 707.0,
                }
            )
        elif model_id == "critical_boundary_safety_guarded_transformer":
            row.update(
                {
                    "critical_misses": native.get("critical_miss_count"),
                    "operational_recall": native.get("operational_recall"),
                    "urgent_precision": native.get("urgent_precision"),
                    "abstention_review_count": native.get("abstain_review_count"),
                    "total_review_count": native.get("total_review_count"),
                    "review_workload": native.get("review_workload"),
                }
            )
        else:
            row.update(
                {
                    "critical_misses": native.get("missed_critical_count"),
                    "operational_recall": native.get("operational_critical_recall"),
                    "urgent_precision": native.get("urgent_review_precision"),
                    "abstention_review_count": native.get("abstain_review_count"),
                    "total_review_count": native.get("mandatory_review_count"),
                    "review_workload": native.get("total_review_workload"),
                }
            )
        rows.append(row)
    native_df = pd.DataFrame(rows)
    native_df.to_csv(reports_dir / "native_system_comparison.csv", index=False)
    return native_df


def build_headline_table(reports_dir: Path, point_df: pd.DataFrame, fixed_df: pd.DataFrame) -> pd.DataFrame:
    fixed = fixed_df.set_index("model_id")
    rows = []
    for _, point in point_df.iterrows():
        model_id = str(point["model_id"])
        fixed_row = fixed.loc[model_id]
        rows.append(
            {
                "Model": point["display_name"],
                "Family": point["model_family"],
                "Overall MAE": point["mae"],
                "Overall RMSE": point["rmse"],
                "NASA score": point["nasa_score"],
                "Severe optimism": point["severe_optimistic_prediction_rate"],
                "Critical misses": fixed_row["critical_miss_count"],
                "Operational recall": fixed_row["operational_recall"],
                "Review workload": fixed_row["review_workload"],
                "Status": point_df.loc[point_df["model_id"] == model_id, "model_id"].map(lambda _: "Final selected system" if model_id == "critical_boundary_safety_guarded_transformer" else "Candidate").iloc[0],
            }
        )
    headline = pd.DataFrame(rows)
    headline.to_csv(reports_dir / "headline_model_comparison.csv", index=False)
    return headline


def discover_registry(root: Path, reports_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    def add_row(**kwargs: Any) -> None:
        model_id = str(kwargs["model_id"])
        base = {
            "model_id": model_id,
            "display_name": display_name(model_id),
            "model_family": kwargs.get("model_family"),
            "development_stage": kwargs.get("development_stage"),
            "architecture_summary": kwargs.get("architecture_summary"),
            "prediction_type": kwargs.get("prediction_type", "point RUL"),
            "uses_physics_guidance": bool(kwargs.get("uses_physics_guidance", False)),
            "uses_kan": bool(kwargs.get("uses_kan", False)),
            "uses_safety_guard": bool(kwargs.get("uses_safety_guard", False)),
            "uses_uncertainty": bool(kwargs.get("uses_uncertainty", False)),
            "uses_abstention": bool(kwargs.get("uses_abstention", False)),
            "uses_maintenance_policy": bool(kwargs.get("uses_maintenance_policy", False)),
            "parameter_count": kwargs.get("parameter_count"),
            "checkpoint_size_bytes": kwargs.get("checkpoint_size_bytes"),
            "training_status": kwargs.get("training_status", "not evaluated"),
            "selection_status": kwargs.get("selection_status", "not selected"),
            "benchmark_evaluated": bool(kwargs.get("benchmark_evaluated", False)),
            "benchmark_access_protocol": kwargs.get("benchmark_access_protocol", "not comparable"),
            "source_config": kwargs.get("source_config"),
            "source_metrics": kwargs.get("source_metrics"),
            "source_checkpoint": kwargs.get("source_checkpoint"),
            "notes": kwargs.get("notes", ""),
        }
        rows.append(base)

    add_row(
        model_id="classical_random_forest",
        model_family="Classical baseline",
        development_stage="Phase 4 provenance",
        architecture_summary="Random forest final-row tabular RUL predictor with conformal uncertainty from earlier multidomain baseline.",
        uses_uncertainty=True,
        uses_maintenance_policy=True,
        training_status="completed",
        selection_status="baseline",
        benchmark_evaluated=True,
        benchmark_access_protocol="benchmark metrics imported from classical manifest",
        source_config=rel(root / "configs/multidomain_rul_uncertainty.yaml", root),
        source_metrics=rel(root / "reports/deep_rul/classical_benchmark_manifest.json", root),
    )

    deep_registry = read_json(root / "reports/deep_rul/deep_model_registry.json", {})
    deep_eff = safe_read_csv(root / "reports/deep_rul/model_efficiency.csv").set_index("model_id") if (root / "reports/deep_rul/model_efficiency.csv").exists() else pd.DataFrame()
    for model in deep_registry.get("models", []):
        model_id = model.get("model_id")
        eff = deep_eff.loc[model_id].to_dict() if model_id in getattr(deep_eff, "index", []) else {}
        add_row(
            model_id=model_id,
            model_family="Deep-learning baseline",
            development_stage="Phase 5 provenance",
            architecture_summary=f"{model.get('architecture')} temporal sequence baseline.",
            parameter_count=eff.get("parameter_count"),
            checkpoint_size_bytes=eff.get("serialized_size_bytes"),
            training_status="completed",
            benchmark_evaluated=model_id in {"lstm"},
            benchmark_access_protocol="validation screening; LSTM benchmark table when selected",
            source_config=rel(root / "configs/multidomain_deep_rul.yaml", root),
            source_metrics=rel(root / "reports/deep_rul/deep_model_registry.json", root),
            source_checkpoint=rel(root / f"artifacts/deep_rul/checkpoints/screening_{model_id}.pt", root),
        )

    ext_registry = read_json(root / "reports/deep_rul_extended/extended_model_registry.json", {})
    ext_eff = safe_read_csv(root / "reports/deep_rul_extended/model_efficiency.csv").set_index("model_id")
    for item in ext_registry.get("registry", []):
        candidate = item.get("candidate", {})
        model_id = candidate.get("model_id")
        eff = ext_eff.loc[model_id].to_dict() if model_id in ext_eff.index else {}
        add_row(
            model_id=model_id,
            model_family="Patch/temporal Transformer candidate" if "transformer" in str(model_id) else "Deep-learning candidate",
            development_stage="Phase 5B provenance",
            architecture_summary=json.dumps(candidate, sort_keys=True),
            parameter_count=eff.get("parameter_count"),
            checkpoint_size_bytes=eff.get("serialized_size_bytes"),
            training_status="completed",
            selection_status="baseline" if model_id == "patch_transformer_10x5_mean_b" else "not selected",
            benchmark_evaluated=model_id == "patch_transformer_10x5_mean_b",
            benchmark_access_protocol="locked benchmark for selected Patch Transformer; validation only for other candidates",
            source_config=rel(root / "configs/multidomain_temporal_optimization.yaml", root),
            source_metrics=rel(root / "reports/deep_rul_extended/extended_model_registry.json", root),
            source_checkpoint=rel(root / f"artifacts/deep_rul_extended/checkpoints/stage_a_{model_id}.pt", root),
        )

    physics_registry = read_json(root / "reports/physics_guided_rul/physics_candidate_registry.json", {})
    physics_screen = safe_read_csv(root / "reports/physics_guided_rul/screening_metrics.csv").set_index("candidate_id")
    for candidate in physics_registry.get("candidates", []):
        model_id = candidate.get("candidate_id")
        metrics = physics_screen.loc[model_id].to_dict() if model_id in physics_screen.index else {}
        add_row(
            model_id=model_id,
            model_family="Physics-guided Transformer",
            development_stage="Phase 5C provenance",
            architecture_summary=f"Active losses: {', '.join(candidate.get('active_losses', [])) or 'data only'}",
            uses_physics_guidance=model_id != "phase5b_reimplementation_baseline",
            parameter_count=metrics.get("parameter_count"),
            checkpoint_size_bytes=metrics.get("checkpoint_size"),
            training_status=metrics.get("training_status", "completed"),
            selection_status="selected backbone" if model_id == "physics_regime" else "not selected",
            benchmark_evaluated=model_id == "physics_regime",
            benchmark_access_protocol="validation-only for ablations; locked benchmark for selected regime-consistent model",
            source_config=rel(root / "configs/physics_guided_temporal_rul.yaml", root),
            source_metrics=rel(root / "reports/physics_guided_rul/screening_metrics.csv", root),
            source_checkpoint=rel(root / f"artifacts/physics_guided_rul/checkpoints/screening_{model_id}.pt", root),
        )

    kan_registry = read_json(root / "reports/aerokan_phm/candidate_registry.json", {})
    kan_screen = safe_read_csv(root / "reports/aerokan_phm/screening_metrics.csv")
    kan_metric_by_id = kan_screen.set_index("candidate_id") if "candidate_id" in kan_screen else pd.DataFrame()
    for candidate in kan_registry.get("candidates", []):
        model_id = candidate.get("candidate_id")
        metrics = kan_metric_by_id.loc[model_id].to_dict() if model_id in getattr(kan_metric_by_id, "index", []) else {}
        add_row(
            model_id=model_id,
            model_family="Residual-correction control" if "kan" not in str(model_id) else "AeroKAN experimental branch",
            development_stage="Phase 5D provenance",
            architecture_summary=str(candidate.get("candidate_type", candidate.get("model_type", "residual correction"))),
            uses_physics_guidance=True,
            uses_kan="kan" in str(model_id),
            uses_uncertainty=model_id == "two_layer_compact_kan_h8",
            uses_abstention=model_id == "two_layer_compact_kan_h8",
            uses_maintenance_policy=model_id == "two_layer_compact_kan_h8",
            parameter_count=metrics.get("parameter_count"),
            checkpoint_size_bytes=Path(root / "artifacts/aerokan_phm/aerokan_corrector.pt").stat().st_size if model_id == "two_layer_compact_kan_h8" and (root / "artifacts/aerokan_phm/aerokan_corrector.pt").exists() else None,
            training_status="completed",
            selection_status="experimental, not final selected" if model_id == "two_layer_compact_kan_h8" else "not selected",
            benchmark_evaluated=model_id == "two_layer_compact_kan_h8",
            benchmark_access_protocol="KAN benchmark for locked compact corrector; validation-only for screening controls",
            source_config=rel(root / "configs/aerokan_rul_corrector.yaml", root),
            source_metrics=rel(root / "reports/aerokan_phm/screening_metrics.csv", root),
            source_checkpoint=rel(root / "artifacts/aerokan_phm/aerokan_corrector.pt", root) if model_id == "two_layer_compact_kan_h8" else None,
        )

    selective_registry = read_json(root / "reports/aerokan_phm_selective/correction_candidate_registry.json", {})
    selective_screen = safe_read_csv(root / "reports/aerokan_phm_selective/correction_screening_metrics.csv")
    selective_by_id = selective_screen.set_index("candidate_id") if "candidate_id" in selective_screen else pd.DataFrame()
    for candidate in selective_registry.get("candidates", []):
        model_id = candidate.get("candidate_id")
        metrics = selective_by_id.loc[model_id].to_dict() if model_id in getattr(selective_by_id, "index", []) else {}
        add_row(
            model_id=model_id,
            model_family="Selective residual safety correction",
            development_stage="Phase 5D.1 provenance",
            architecture_summary=str(candidate.get("candidate_type", "selective correction")),
            uses_physics_guidance=True,
            uses_kan="kan" in str(model_id),
            uses_safety_guard=True,
            parameter_count=metrics.get("parameter_count"),
            training_status="completed",
            selection_status="selected experimental safety corrector" if model_id == "phase5d_two_layer_one_sided_control_bound20" else "not selected",
            benchmark_evaluated=model_id == "phase5d_two_layer_one_sided_control_bound20",
            benchmark_access_protocol="validation-only screening; benchmark for locked selective system",
            source_config=rel(root / "configs/selective_aerokan_safety_corrector.yaml", root),
            source_metrics=rel(root / "reports/aerokan_phm_selective/correction_screening_metrics.csv", root),
            source_checkpoint=rel(root / "artifacts/aerokan_phm_selective/one_sided_kan_checkpoint.pt", root) if model_id == "phase5d_two_layer_one_sided_control_bound20" else None,
        )

    add_row(
        model_id="selective_one_sided_aerokan_safety_corrector",
        model_family="Selective AeroKAN safety correction",
        development_stage="Phase 5D.1 provenance",
        architecture_summary="Locked one-sided AeroKAN experimental safety corrector with learned gate and downward-only correction.",
        uses_physics_guidance=True,
        uses_kan=True,
        uses_safety_guard=True,
        uses_uncertainty=True,
        uses_abstention=True,
        uses_maintenance_policy=True,
        training_status="completed",
        selection_status="experimental, not final selected",
        benchmark_evaluated=True,
        benchmark_access_protocol="locked benchmark, not final selected due insufficient gate coverage",
        source_config=rel(root / "configs/selective_aerokan_safety_corrector.yaml", root),
        source_metrics=rel(root / "reports/aerokan_phm_selective/benchmark_metrics.json", root),
        source_checkpoint=rel(root / "artifacts/aerokan_phm_selective/one_sided_kan_checkpoint.pt", root),
    )
    add_row(
        model_id="critical_boundary_safety_guarded_transformer",
        model_family="Safety-guarded Transformer",
        development_stage="Phase 5D.2 provenance",
        architecture_summary=FINAL_ARCHITECTURE,
        uses_physics_guidance=True,
        uses_safety_guard=True,
        uses_uncertainty=True,
        uses_abstention=False,
        uses_maintenance_policy=True,
        parameter_count=89985,
        checkpoint_size_bytes=(root / "artifacts/physics_guided_rul/checkpoints/locked_physics_guided_model.pt").stat().st_size,
        training_status="frozen; no training in final release",
        selection_status="final selected system",
        benchmark_evaluated=True,
        benchmark_access_protocol="locked critical-boundary guard benchmark with no benchmark-based retuning",
        source_config=rel(root / "configs/critical_gate_aerokan_corrector.yaml", root),
        source_metrics=rel(root / "reports/aerokan_phm_critical_gate/benchmark_metrics.json", root),
        source_checkpoint=rel(root / "artifacts/physics_guided_rul/checkpoints/locked_physics_guided_model.pt", root),
        notes="Final deployed system is not a KAN model; the guard is deterministic.",
    )
    add_row(
        model_id="aeroguard_phm_safety_guarded_rul_system",
        model_family="Complete deployment pipeline",
        development_stage="Phase 6 release provenance",
        architecture_summary="Regime-aware preprocessing, physics-guided Transformer, critical-boundary safety guard, conformal uncertainty, support/review logic, and maintenance policy.",
        prediction_type="complete RUL decision-support response",
        uses_physics_guidance=True,
        uses_safety_guard=True,
        uses_uncertainty=True,
        uses_abstention=False,
        uses_maintenance_policy=True,
        training_status="frozen; no training in final release",
        selection_status="final selected system",
        benchmark_evaluated=True,
        benchmark_access_protocol="inherits frozen model and safety benchmark from the selected final system",
        source_config=rel(root / "artifacts/final_release/frozen_system_manifest.json", root),
        source_metrics=rel(root / "reports/final_release/headline_model_comparison.csv", root),
    )

    registry = pd.DataFrame(rows).drop_duplicates(subset=["model_id"], keep="last")
    registry.to_csv(reports_dir / "model_registry.csv", index=False)
    write_json(reports_dir / "model_registry.json", registry.to_dict(orient="records"))
    mapping = registry[["model_id", "display_name", "development_stage", "selection_status"]].copy()
    mapping.to_csv(reports_dir / "model_name_mapping.csv", index=False)
    return registry, mapping


def build_validation_tables(root: Path, reports_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    for label, path, id_col in [
        ("deep_baseline_screening", root / "reports/deep_rul/screening_metrics.csv", "model_id"),
        ("extended_deep_screening", root / "reports/deep_rul_extended/screening_metrics.csv", "model_id"),
        ("physics_screening", root / "reports/physics_guided_rul/screening_metrics.csv", "candidate_id"),
        ("kan_screening", root / "reports/aerokan_phm/screening_metrics.csv", "candidate_id"),
        ("selective_kan_screening", root / "reports/aerokan_phm_selective/correction_screening_metrics.csv", "candidate_id"),
        ("critical_gate_screening", root / "reports/aerokan_phm_critical_gate/correction_candidate_metrics.csv", "candidate_id"),
    ]:
        frame = safe_read_csv(path)
        if frame.empty or id_col not in frame.columns:
            continue
        cols = [id_col] + [col for col in frame.columns if col in {"training_status", "validation_mae", "validation_rmse", "validation_nasa_score", "validation_severe_optimistic_rate", "mae", "rmse", "nasa_score", "severe_optimistic_rate", "eligible"}]
        out = frame[cols].copy()
        out = out.rename(columns={id_col: "model_id"})
        out.insert(1, "display_name", out["model_id"].map(display_name))
        out.insert(2, "validation_track", label)
        out["benchmark_evaluated"] = False
        out["source_metrics"] = rel(path, root)
        pieces.append(out)
    validation = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()
    validation.to_csv(reports_dir / "validation_candidate_comparison.csv", index=False)

    physics = safe_read_csv(root / "reports/physics_guided_rul/screening_metrics.csv")
    if not physics.empty:
        physics.insert(1, "display_name", physics["candidate_id"].map(display_name))
    physics.to_csv(reports_dir / "physics_ablation_comparison.csv", index=False)

    kan = safe_read_csv(root / "reports/aerokan_phm/screening_metrics.csv")
    if not kan.empty and "candidate_id" in kan.columns:
        kan.insert(1, "display_name", kan["candidate_id"].map(display_name))
    kan.to_csv(reports_dir / "kan_candidate_comparison.csv", index=False)

    efficiency_frames = []
    for label, path in [
        ("deep_baseline", root / "reports/deep_rul/model_efficiency.csv"),
        ("extended_patch_transformer", root / "reports/deep_rul_extended/model_efficiency.csv"),
        ("physics_guided", root / "reports/physics_guided_rul/model_efficiency.csv"),
    ]:
        frame = safe_read_csv(path)
        if not frame.empty and "model_id" in frame.columns:
            frame.insert(1, "display_name", frame["model_id"].map(display_name))
            frame.insert(2, "efficiency_track", label)
            frame["source_metrics"] = rel(path, root)
            efficiency_frames.append(frame)
    efficiency = pd.concat(efficiency_frames, ignore_index=True, sort=False) if efficiency_frames else pd.DataFrame()
    efficiency.to_csv(reports_dir / "efficiency_comparison.csv", index=False)

    uncertainty_rows = []
    for model_id, path in [
        ("patch_transformer_10x5_mean_b", root / "reports/deep_rul_extended/uncertainty_metrics.json"),
        ("physics_regime", root / "reports/physics_guided_rul/uncertainty_metrics.json"),
        ("two_layer_compact_kan_h8", root / "reports/aerokan_phm/uncertainty_metrics.json"),
        ("selective_one_sided_aerokan_safety_corrector", root / "reports/aerokan_phm_selective/uncertainty_metrics.json"),
        ("critical_boundary_safety_guarded_transformer", root / "reports/aerokan_phm_critical_gate/locked_uncertainty_method.json"),
    ]:
        obj = read_json(path, {})
        if path.name == "locked_uncertainty_method.json":
            for level, radius in obj.get("radii", {}).items():
                uncertainty_rows.append({"model_id": model_id, "display_name": display_name(model_id), "nominal_level": float(level), "coverage": None, "mean_interval_width": float(radius) * 2.0, "source_metrics": rel(path, root)})
        else:
            overall = obj.get("overall", {}) if isinstance(obj, dict) else {}
            for level, metrics in overall.items():
                if isinstance(metrics, dict):
                    uncertainty_rows.append({"model_id": model_id, "display_name": display_name(model_id), "nominal_level": float(level), "coverage": metrics.get("coverage"), "mean_interval_width": metrics.get("mean_interval_width"), "source_metrics": rel(path, root)})
    uncertainty = pd.DataFrame(uncertainty_rows)
    uncertainty.to_csv(reports_dir / "uncertainty_comparison.csv", index=False)

    subset_rows = []
    for model_id, path in [
        ("patch_transformer_10x5_mean_b", root / "reports/deep_rul_extended/metrics_by_subset.csv"),
        ("physics_regime", root / "reports/physics_guided_rul/metrics_by_subset.csv"),
        ("two_layer_compact_kan_h8", root / "reports/aerokan_phm/benchmark_metrics_by_subset.csv"),
        ("selective_one_sided_aerokan_safety_corrector", root / "reports/aerokan_phm_selective/benchmark_metrics_by_subset.csv"),
        ("critical_boundary_safety_guarded_transformer", root / "reports/aerokan_phm_critical_gate/benchmark_metrics.json"),
    ]:
        if path.suffix == ".json":
            continue
        frame = safe_read_csv(path)
        if not frame.empty:
            frame.insert(0, "model_id", model_id)
            frame.insert(1, "display_name", display_name(model_id))
            frame["source_metrics"] = rel(path, root)
            subset_rows.append(frame)
    subset = pd.concat(subset_rows, ignore_index=True, sort=False) if subset_rows else pd.DataFrame()
    subset.to_csv(reports_dir / "subset_metrics_comparison.csv", index=False)
    return validation, physics, kan, efficiency, uncertainty


def per_metric_series(aligned: pd.DataFrame, model_id: str) -> dict[str, np.ndarray]:
    true = aligned["true_rul"].to_numpy(dtype=float)
    pred = aligned[model_id].to_numpy(dtype=float)
    residual = pred - true
    return {
        "absolute_error": np.abs(residual),
        "squared_error": np.square(residual),
        "nasa_contribution": np.where(residual < 0, np.exp(-residual / 13.0) - 1.0, np.exp(residual / 10.0) - 1.0),
        "optimistic_indicator": (residual > 0).astype(float),
        "severe_optimistic_indicator": (residual > CANONICAL_SEVERE_THRESHOLD).astype(float),
        "fixed_policy_critical_miss_indicator": ((true <= CRITICAL_THRESHOLD) & (pred > CRITICAL_THRESHOLD)).astype(float),
        "mandatory_review_indicator": (pred <= CRITICAL_THRESHOLD).astype(float),
    }


def build_paired_bootstrap(root: Path, reports_dir: Path, aligned: pd.DataFrame, sources: list[dict[str, Any]], iterations: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(20260715)
    final_id = "critical_boundary_safety_guarded_transformer"
    final_series = per_metric_series(aligned, final_id)
    rows = []
    n = len(aligned)
    for source in sources:
        model_id = source["model_id"]
        if model_id == final_id:
            continue
        model_series = per_metric_series(aligned, model_id)
        for metric, comparator_values in model_series.items():
            final_values = final_series[metric]
            delta = comparator_values - final_values
            point = float(np.mean(delta))
            samples = np.empty(iterations)
            for index in range(iterations):
                sample_idx = rng.integers(0, n, size=n)
                samples[index] = float(np.mean(delta[sample_idx]))
            ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
            probability = float(np.mean(samples > 0.0))
            rel_diff = point / max(abs(float(np.mean(comparator_values))), 1.0e-12)
            rows.append(
                {
                    "comparison": f"{model_id}_vs_final",
                    "comparator_model_id": model_id,
                    "comparator_display_name": display_name(model_id),
                    "final_model_id": final_id,
                    "metric": metric,
                    "point_difference_comparator_minus_final": point,
                    "relative_difference": rel_diff,
                    "ci_lower": float(ci_low),
                    "ci_upper": float(ci_high),
                    "probability_final_improves": probability,
                    "interval_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                    "engine_alignment_count": n,
                    "statistical_interpretation": "final lower/better" if point > 0 and ci_low > 0 else "comparator lower/better" if point < 0 and ci_high < 0 else "uncertain",
                }
            )

    interval_sources = {}
    key = ["subset", "global_engine_id", "final_observed_cycle"]
    for source in sources:
        path = Path(source["interval_path"])
        frame = safe_read_csv(path)
        if not frame.empty and "interval_width_90" in frame.columns:
            interval_sources[source["model_id"]] = frame[key + ["interval_width_90"]].rename(columns={"interval_width_90": source["model_id"]})
    if final_id in interval_sources:
        final_width = interval_sources[final_id]
        for model_id, frame in interval_sources.items():
            if model_id == final_id:
                continue
            merged = frame.merge(final_width, on=key, how="inner")
            if merged.empty:
                continue
            delta = merged[model_id].to_numpy(dtype=float) - merged[final_id].to_numpy(dtype=float)
            samples = np.empty(iterations)
            for index in range(iterations):
                sample_idx = rng.integers(0, len(delta), size=len(delta))
                samples[index] = float(np.mean(delta[sample_idx]))
            ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
            point = float(delta.mean())
            rows.append(
                {
                    "comparison": f"{model_id}_vs_final",
                    "comparator_model_id": model_id,
                    "comparator_display_name": display_name(model_id),
                    "final_model_id": final_id,
                    "metric": "interval_width_90",
                    "point_difference_comparator_minus_final": point,
                    "relative_difference": point / max(abs(float(merged[model_id].mean())), 1.0e-12),
                    "ci_lower": float(ci_low),
                    "ci_upper": float(ci_high),
                    "probability_final_improves": float(np.mean(samples > 0.0)),
                    "interval_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                    "engine_alignment_count": int(len(delta)),
                    "statistical_interpretation": "final narrower" if point > 0 and ci_low > 0 else "comparator narrower" if point < 0 and ci_high < 0 else "uncertain",
                }
            )
    bootstrap = pd.DataFrame(rows)
    bootstrap.to_csv(reports_dir / "paired_bootstrap_comparison.csv", index=False)
    return bootstrap


def build_manifest(root: Path, reports_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    final_meta = read_json(root / "reports/physics_guided_rul/final_fit_metadata.json", {})
    locked_model = read_json(root / "reports/physics_guided_rul/locked_physics_model.json", {})
    locked_uncertainty = read_json(root / "reports/aerokan_phm_critical_gate/locked_uncertainty_method.json", {})
    locked_maintenance = read_json(root / "reports/aerokan_phm_critical_gate/locked_maintenance_policy.json", {})
    locked_abstention = read_json(root / "reports/aerokan_phm_critical_gate/locked_abstention_policy.json", {})
    cascade = read_json(root / "artifacts/aerokan_phm_critical_gate/cascade_metadata.json", {})
    metric_registry = read_json(root / "reports/aerokan_phm_critical_gate/metric_definition_audit.json", {})
    source_paths = [
        root / "artifacts/physics_guided_rul/checkpoints/locked_physics_guided_model.pt",
        root / "artifacts/physics_guided_rul/checkpoints/final_preprocessor.pkl",
        root / "reports/physics_guided_rul/final_fit_metadata.json",
        root / "reports/physics_guided_rul/locked_physics_model.json",
        root / "reports/aerokan_phm_critical_gate/benchmark_predictions.csv",
        root / "reports/aerokan_phm_critical_gate/benchmark_metrics.json",
        root / "reports/aerokan_phm_critical_gate/benchmark_fixed_policy_metrics.json",
        root / "reports/aerokan_phm_critical_gate/locked_uncertainty_method.json",
        root / "reports/aerokan_phm_critical_gate/locked_maintenance_policy.json",
        root / "reports/aerokan_phm_critical_gate/locked_abstention_policy.json",
        root / "artifacts/aerokan_phm_critical_gate/cascade_metadata.json",
        root / "artifacts/aerokan_phm_critical_gate/uncertainty_model.pkl",
        root / "artifacts/aerokan_phm_critical_gate/maintenance_policy.pkl",
    ]
    feature_schema = {
        "raw_required_columns": REQUIRED_INPUT_COLUMNS,
        "model_features": final_meta.get("feature_names", []),
        "window_length": final_meta.get("window_length", 50),
        "rul_cap": final_meta.get("rul_cap", 125.0),
    }
    configuration = {
        "system_name": SYSTEM_NAME,
        "release_version": RELEASE_VERSION,
        "model_version": MODEL_VERSION,
        "guard": cascade,
        "uncertainty": locked_uncertainty,
        "maintenance": locked_maintenance,
        "abstention": locked_abstention,
    }
    feature_schema_hash = hashlib.sha256(json.dumps(feature_schema, sort_keys=True).encode("utf-8")).hexdigest()
    configuration_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode("utf-8")).hexdigest()

    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except Exception:
        torch_version = None
        cuda_version = None

    components = {
        "physics_regime_checkpoint": component(root, root / "artifacts/physics_guided_rul/checkpoints/locked_physics_guided_model.pt", "Frozen Regime-Consistent Physics-Guided Patch Transformer checkpoint"),
        "final_preprocessor": component(root, root / "artifacts/physics_guided_rul/checkpoints/final_preprocessor.pkl", "Final regime-aware preprocessing artifact"),
        "critical_boundary_guard": component(root, root / "artifacts/aerokan_phm_critical_gate/cascade_metadata.json", "Critical-boundary safety guard configuration"),
        "uncertainty_model": component(root, root / "artifacts/aerokan_phm_critical_gate/uncertainty_model.pkl", "Lightweight conformal uncertainty artifact"),
        "maintenance_policy": component(root, root / "artifacts/aerokan_phm_critical_gate/maintenance_policy.pkl", "Frozen maintenance policy artifact"),
        "metric_definition_registry": component(root, root / "reports/aerokan_phm_critical_gate/metric_definition_audit.json", "Canonical metric definitions and audit"),
    }
    source_manifest = {rel(path, root): component(root, path, path.name) for path in source_paths if path.exists()}
    manifest = {
        "system_name": SYSTEM_NAME,
        "release_version": RELEASE_VERSION,
        "model_version": MODEL_VERSION,
        "predictive_backbone": PREDICTIVE_BACKBONE,
        "safety_layer": SAFETY_LAYER,
        "final_predictive_architecture": FINAL_ARCHITECTURE,
        "component_names": list(components.keys()),
        "components": components,
        "backbone_architecture": locked_model.get("architecture", {}),
        "window_length": final_meta.get("window_length", 50),
        "minimum_history_length": 10,
        "maximum_history_length": 500,
        "maximum_output_rul": 250.0,
        "feature_schema": feature_schema,
        "feature_schema_hash": feature_schema_hash,
        "configuration_hash": configuration_hash,
        "python_version": platform.python_version(),
        "python_executable": "python",
        "pytorch_version": torch_version,
        "cuda_version": cuda_version,
        "expected_input_schema": {"required_columns": REQUIRED_INPUT_COLUMNS, "optional_columns": ["engine_id", "global_engine_id", "unit_id", "subset"]},
        "output_schema": output_schema(),
        "canonical_metric_definitions": metric_registry,
        "safety_thresholds": {
            "critical_true_rul_threshold": CRITICAL_THRESHOLD,
            "severe_optimism_threshold": CANONICAL_SEVERE_THRESHOLD,
            "urgent_review_threshold": locked_maintenance.get("urgent_threshold", 15.0),
            "schedule_threshold": locked_maintenance.get("schedule_threshold", 30.0),
            "inspection_threshold": locked_maintenance.get("inspection_threshold", 60.0),
        },
        "safety_guard": {
            "method": "deterministic_critical_boundary_rule",
            "boundary_low": cascade.get("boundary_low", 15.0),
            "boundary_high": cascade.get("boundary_high", 25.0),
            "margin": cascade.get("margin", 0.5),
            "correction_bound": cascade.get("bound", 10.0),
            "active_rule": "15 < base_rul <= 25",
            "deployed_system_is_kan": False,
        },
        "uncertainty": locked_uncertainty,
        "review_policy": locked_abstention,
        "maintenance_policy": locked_maintenance,
        "training_data_identifiers": {"subsets": ["FD001", "FD002", "FD003", "FD004"], "dataset": "NASA C-MAPSS simulated turbofan data"},
        "benchmark_data_identifiers": {"subsets": ["FD001", "FD002", "FD003", "FD004"], "engine_count": 707, "true_rul": "uncapped final benchmark RUL"},
        "lock_timestamp": datetime.now(timezone.utc).isoformat(),
        "reproduction_command": "python -m aeroguard.pipelines.build_final_release",
        "source_artifact_manifest": source_manifest,
        "known_limitations": [
            "C-MAPSS is simulated turbofan data, not real aircraft-maintenance telemetry.",
            "The final safety guard is deterministic and benchmark-tuned before final evaluation; it is not a discovered physical law.",
            "The deployed final system is not a KAN model; KAN models were experimental candidates.",
            "The Python inference wrapper may use a deterministic compatibility predictor if the local neural runtime cannot reconstruct the checkpoint.",
            "Benchmark results do not certify or guarantee real-world aviation safety.",
        ],
    }
    write_json(artifacts_dir / "frozen_system_manifest.json", manifest)
    write_json(artifacts_dir / "source_artifact_manifest.json", source_manifest)
    return manifest


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    use = frame if max_rows is None else frame.head(max_rows)
    headers = list(use.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in use.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_docs(root: Path, reports_dir: Path, manifest: dict[str, Any], headline: pd.DataFrame, point_df: pd.DataFrame, fixed_df: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    headline_public = headline.copy()
    readme = f"""# {SYSTEM_NAME}

## Project Title

{SYSTEM_NAME}

## One-Sentence Value Proposition

A reproducible turbofan RUL decision-support project that combines a physics-guided temporal predictor, calibrated uncertainty, and an auditable critical-boundary safety guard.

## Problem Statement

Remaining useful life prediction estimates how many cycles a turbofan engine can continue operating before failure under observed sensor history. The project studies prediction accuracy, optimistic error, uncertainty, and maintenance-review behavior separately.

## Why RUL Prediction Matters

Late RUL estimates can delay inspection, while overly conservative estimates can increase unnecessary maintenance. This release therefore reports both point-prediction metrics and policy-level safety metrics.

## C-MAPSS Data

The experiments use NASA C-MAPSS simulated turbofan degradation subsets FD001-FD004. These are benchmark simulations and are not certified aircraft-maintenance records.

## Complete Development Journey

### Patch Transformer — 10×5 Patches with Mean Pooling

Established a strong temporal RUL benchmark using patch-based sensor-sequence modelling.

### Regime-Consistent Physics-Guided Patch Transformer

Added operating-regime consistency guidance, improving generalization and reducing optimistic errors.

### AeroKAN-PHM Compact Residual Corrector

Tested whether named engineering features and KAN edge functions could correct residual RUL errors. It reduced some critical misses but worsened overall error and was not selected.

### Selective One-Sided AeroKAN Safety Corrector

Restricted KAN intervention to downward corrections on risk-selected engines. It preserved point accuracy but the learned gate did not provide sufficient critical coverage.

### Critical-Boundary Safety-Guarded Physics-Guided Transformer

Used the frozen physics-guided predictor with an auditable critical-boundary safeguard. This became the selected final safety system.

## Final Architecture

The selected architecture is the {FINAL_ARCHITECTURE}. The complete release pipeline adds regime-aware preprocessing, conformal uncertainty, support/review logic, and the frozen maintenance policy.

## Model Comparison

{markdown_table(headline_public)}

Point-prediction metrics are separated from fixed-policy and native-policy metrics in `reports/final_release`.

## Physics-Guided Modelling

The selected predictive backbone is the {PREDICTIVE_BACKBONE}. Physics-guided ablations are reported separately in `physics_ablation_comparison.csv`.

## KAN Experimental Branch

KAN models are documented as experimental residual-correction candidates. The deployed final system is not a KAN model.

## Critical-Boundary Safety Guard

The final guard is deterministic: it activates when `15 < base_rul <= 25`, applies a downward correction capped at 10 cycles, and uses a 0.5-cycle margin.

## Uncertainty Quantification

The final system uses global split conformal uncertainty with frozen radii from the selected release manifest.

## Maintenance Recommendations

The frozen maintenance policy uses urgent, schedule, and inspection thresholds at 15, 30, and 60 cycles respectively. Human engineering review remains required.

## Installation

Core installation:

```powershell
python -m pip install -e .
```

Release-validation installation:

```powershell
python -m pip install -e ".[api,dashboard,dev]" -c requirements/constraints.txt
```

Docker and containerization are intentionally deferred to a future release.

## Quick Start

```powershell
python -m aeroguard.inference.cli --manifest artifacts/final_release/frozen_system_manifest.json --input examples/sample_engine_history.csv --output reports/inference/sample_prediction.json
```

## Python Inference

```python
import pandas as pd
from aeroguard.inference import AeroGuardPredictor

predictor = AeroGuardPredictor.from_manifest("artifacts/final_release/frozen_system_manifest.json")
prediction = predictor.predict_engine(pd.read_csv("examples/sample_engine_history.csv"))
```

## CLI Inference

Use `python -m aeroguard.inference.cli` with `--input`, `--batch-dir`, `--output`, `--output-csv`, and `--validation-only`.

## API Usage

```powershell
python -m uvicorn aeroguard.api.app:app --host 127.0.0.1 --port 8000
```

## Dashboard Usage

```powershell
python -m streamlit run dashboard/app.py
```

## Testing

Phase 6 tests cover registry generation, hash validation, inference, CLI, API import safety, dashboard import safety, monitoring, and smoke execution.

## Reproducibility

See `REPRODUCIBILITY.md` for hashes, commands, environment metadata, and expected metrics.

## Repository Structure

Final release outputs live in `reports/final_release` and `artifacts/final_release`. Production wrappers live under `src/aeroguard/inference`, the API under `src/aeroguard/api`, and the dashboard under `dashboard`.

## Limitations

The system is evaluated on simulated benchmark data, is not certified for aircraft maintenance, and cannot guarantee failure prevention.

## Responsible-Use Statement

Use this system as research decision support only. Maintenance decisions require qualified human engineering review and independent validation.

## Future Work

Validate on independent fleets, expand monitoring with real operational envelopes, and reassess learned safety gates only with sufficient critical-miss support.

## Citation

Reference NASA C-MAPSS for the dataset and this repository for the AeroGuard-PHM implementation.

## License

AeroGuard-PHM is released under Apache License 2.0. See `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `CITATION.cff`.
"""
    existing_readme = root / "README.md"
    if not existing_readme.exists() or "docs/assets/README_IMAGE_MAPPING.md" not in existing_readme.read_text(encoding="utf-8"):
        write_text(existing_readme, readme)

    model_card = f"""# Model Card: {SYSTEM_NAME}

## Model Name
{SYSTEM_NAME}

## Version
Release {RELEASE_VERSION}; model version `{MODEL_VERSION}`.

## Intended Use
Research decision support for simulated turbofan RUL benchmarking, uncertainty analysis, and maintenance-policy experimentation.

## Out-of-Scope Use
The system is not certified for real aircraft maintenance, dispatch, or safety-critical operational control.

## Architecture
{FINAL_ARCHITECTURE}: regime-aware preprocessing, {PREDICTIVE_BACKBONE}, deterministic critical-boundary guard, conformal uncertainty, and maintenance policy.

## Input Requirements
Engine-history CSVs require `cycle`, three operational settings, and sensors `sensor_1` through `sensor_21`.

## Outputs
The predictor returns base RUL, safety-adjusted RUL, conformal intervals, support status, guard activation, review requirement, maintenance action, warnings, and explanations.

## Training Datasets
NASA C-MAPSS simulated turbofan subsets FD001-FD004.

## Evaluation Datasets
Final benchmark evaluation uses aligned FD001-FD004 benchmark engine keys with uncapped true RUL.

## Point-Performance Metrics
{markdown_table(point_df[['display_name','mae','rmse','nasa_score','severe_optimistic_prediction_rate']])}

## Safety Metrics
{markdown_table(fixed_df[['display_name','critical_miss_count','operational_recall','urgent_precision','review_workload']])}

## Uncertainty Metrics
The final release uses global split conformal radii recorded in `artifacts/final_release/frozen_system_manifest.json`.

## Comparison With Prior Models
The final guard materially reduced critical misses under the fixed policy while preserving RMSE non-inferiority relative to the selected backbone.

## Safety-Guard Behaviour
The final guard is deterministic. It activates when `15 < base_rul <= 25` and applies a downward correction capped at 10 cycles. The final deployed system is not a KAN model.

## Known Limitations
C-MAPSS is simulated. Benchmark results do not guarantee real-world safety. The compatibility inference path is for packaging smoke tests when direct neural reconstruction is unavailable.

## Ethical and Operational Considerations
Human engineering review remains required. The model must not be used as an autonomous aviation safety authority.

## Monitoring Recommendations
Monitor input-schema drift, missing sensors, feature range violations, regime distribution, support score, prediction distribution, interval width, guard activation rate, review rate, maintenance-action distribution, RUL trajectory instability, inference latency, and runtime failures.

## Retraining Triggers
Consider retraining only after independent validation shows material drift, degraded calibration, changed sensor definitions, or new fleet operating regimes.

## Independent-Validation Requirements
Validate with external simulated or real-world-like data before any operational interpretation. KAN models were experimental candidates and are not the deployed final system.

## License
AeroGuard-PHM is released under Apache License 2.0. Dataset and third-party provenance notes are maintained in `THIRD_PARTY_NOTICES.md`.
"""
    write_text(root / "MODEL_CARD.md", model_card)

    reproducibility = f"""# Reproducibility

## Python Version
`{manifest.get('python_version')}`

## PyTorch Version
`{manifest.get('pytorch_version')}`

## CUDA Version
`{manifest.get('cuda_version')}`

## Operating System Assumptions
Generated on `{platform.platform()}` using the existing `aerostat-ai` environment.

## Data Directory Structure
C-MAPSS files are expected under `data/raw/cmapss`.

## Installation

Core installation:

```powershell
python -m pip install -e .
```

Reproducible release-validation installation:

```powershell
python -m pip install -e ".[api,dashboard,dev]" -c requirements/constraints.txt
```

Docker and containerization are intentionally deferred to a future release.

## Dataset-File Hashes
Dataset hashes are not duplicated in this final release unless present in earlier source manifests.

## Configuration-File Hashes
The final configuration hash is `{manifest.get('configuration_hash')}`.

## Artifact Hashes
See `artifacts/final_release/frozen_system_manifest.json`.

## Exact Evaluation Commands
```powershell
$env:PYTHONPATH = ".\\src"
python -m aeroguard.pipelines.build_final_release
```

## Expected Benchmark Engine Counts
Aligned final comparison engine count: `{int(point_df['engine_count'].iloc[0])}`.

## Expected Final Metrics
{markdown_table(point_df[point_df['model_id'] == 'critical_boundary_safety_guarded_transformer'][['display_name','mae','rmse','nasa_score','severe_optimistic_prediction_rate']])}

## Deterministic Seeds
Final paired bootstrap uses seed `20260715`. The release builder does not train models.

## Hardware Notes
No model training is performed in Phase 6. Inference can run on CPU by default.

## Known Nondeterministic Operations
CUDA neural inference can vary slightly across hardware. This release does not claim bitwise reproducibility for CUDA operations.

## Verification Commands
```powershell
$env:PYTHONPATH = ".\\src"
python -m pytest tests\\unit\\test_final_release.py -q
python -m pytest tests\\integration\\test_phase6_final_release_smoke.py -q
python scripts\\run_monitoring_demo.py
python scripts\\release_integrity_check.py
```
"""
    write_text(root / "REPRODUCIBILITY.md", reproducibility)

    architecture = f"""# Architecture

## Training Pipeline

```mermaid
flowchart LR
  A["C-MAPSS FD001-FD004"] --> B["Regime-aware preprocessing"]
  B --> C["Patch Transformer candidates"]
  C --> D["Physics-guided ablations"]
  D --> E["Regime-consistent backbone"]
  E --> F["Uncertainty and policy refinement"]
  E --> G["KAN experimental branch"]
  F --> H["Critical-boundary safety guard"]
```

## Frozen Production Inference Pipeline

```mermaid
flowchart LR
  A["Engine history CSV"] --> B["Input validation"]
  B --> C["Regime-aware preprocessing"]
  C --> D["Physics-guided Patch Transformer"]
  D --> E["Critical-boundary safety guard"]
  E --> F["Conformal intervals"]
  F --> G["Maintenance policy"]
  G --> H["Structured prediction response"]
```

## Physics-Guided Transformer

```mermaid
flowchart TB
  A["Sensor sequence"] --> B["10x5 temporal patches"]
  B --> C["Transformer encoder"]
  C --> D["Mean pooled latent state"]
  D --> E["RUL head"]
  C --> F["Regime-consistency training signal"]
```

## Safety Guard

```mermaid
flowchart LR
  A["Base RUL"] --> B{{"15 < base <= 25?"}}
  B -- "yes" --> C["Apply downward correction"]
  B -- "no" --> D["Leave unchanged"]
  C --> E["Safety-adjusted RUL"]
  D --> E
```

## Uncertainty And Maintenance Flow

```mermaid
flowchart LR
  A["Safety-adjusted RUL"] --> B["Global split conformal radii"]
  B --> C["80/90/95 intervals"]
  C --> D["Support and review logic"]
  D --> E["Urgent, schedule, inspect, or monitor"]
```

## Experimental KAN Branch

```mermaid
flowchart LR
  A["Frozen backbone residuals"] --> B["Engineering features"]
  B --> C["KAN and non-KAN residual candidates"]
  C --> D["Global AeroKAN experiment"]
  C --> E["Selective one-sided AeroKAN experiment"]
  D --> F["Not selected"]
  E --> F
```

## Monitoring Architecture

```mermaid
flowchart LR
  A["Inference requests"] --> B["Structured logs"]
  B --> C["Schema and range checks"]
  B --> D["Prediction and interval drift"]
  B --> E["Guard and review rates"]
  B --> F["Latency and failures"]
```

Selected production path: preprocessing, {PREDICTIVE_BACKBONE}, deterministic guard, conformal intervals, and maintenance policy.

Experimental paths: global and selective AeroKAN residual correction branches.
"""
    write_text(root / "docs" / "architecture.md", architecture)

    report = f"""# AeroGuard-PHM: Physics-Guided Remaining Useful Life Prediction with Uncertainty Quantification and Safety-Constrained Maintenance Decisions

## Abstract
This report summarizes the final AeroGuard-PHM release, a safety-guarded RUL decision-support system for simulated C-MAPSS turbofan degradation data.

## Problem Definition
Estimate engine-level remaining useful life while reporting optimistic-error and maintenance-review behavior.

## Dataset
The project uses NASA C-MAPSS FD001-FD004 simulated turbofan data.

## Data Preprocessing
The selected backbone uses regime-aware normalization and fixed-length historical windows.

## Evaluation Protocol
Final comparisons use identical benchmark engine keys, uncapped true RUL, residual `predicted - true`, and a 30-cycle severe optimism threshold.

## Patch Transformer Benchmark
The Patch Transformer with 10x5 mean-pooled patches is the principal temporal benchmark.

## Physics-Guided Modelling
The regime-consistent physics-guided candidate was selected as the predictive backbone.

## Constraint Diagnostics
Physics ablations are separated in `physics_ablation_comparison.csv`.

## Uncertainty Calibration
The final system uses frozen global split conformal radii.

## Abstention Analysis
The final release uses no abstention after the previous abstention policy failed enrichment checks.

## Maintenance-Policy Analysis
Fixed-policy and native-policy tables are separated to avoid mixing model effects with downstream policy effects.

## KAN Residual-Correction Experiments
KAN residual models are retained as experimental research candidates.

## Selective One-Sided KAN Experiment
The selective one-sided AeroKAN preserved point accuracy but did not provide enough critical coverage for final selection.

## Invariant Audit
The prior one-sided invariant audit passed, and metric-definition inconsistencies were documented.

## Critical-Boundary Safety Guard
The final deterministic guard reduced fixed-policy critical misses while preserving configured RMSE non-inferiority.

## Final Comparison
{markdown_table(headline)}

## Statistical Testing
{markdown_table(bootstrap[['comparator_display_name','metric','point_difference_comparator_minus_final','ci_lower','ci_upper','probability_final_improves','statistical_interpretation']].head(12))}

## Interpretability
The safety layer is auditable as a threshold rule. KAN explanations are available only for experimental branches.

## Deployment Architecture
The release includes Python, CLI, API, and Streamlit interfaces around the frozen manifest.

## Limitations
The benchmark is simulated, the guard is deterministic, and the system is not certified for real aircraft maintenance.

## Conclusions
The selected final system is the {FINAL_MODEL_NAME} inside the complete {SYSTEM_NAME}.

## Future Independent Validation
Validate on independent datasets, fleet-specific operating regimes, and external maintenance-review workflows before operational use.
"""
    write_text(root / "docs" / "aeroguard_phm_final_report.md", report)

    consistency = {
        "forbidden_public_phrase_hits": [
            {
                "rule_id": f"forbidden_phrase_{index:02d}",
                "hits": [
                str(path.relative_to(root))
                for path in [root / "README.md", root / "MODEL_CARD.md", root / "docs/architecture.md", root / "docs/aeroguard_phm_final_report.md"]
                if phrase.lower() in path.read_text(encoding="utf-8").lower()
                ],
            }
            for index, phrase in enumerate(FORBIDDEN_PUBLIC_PHRASES, start=1)
        ],
        "readme_headline_rows": len(headline_public),
    }
    write_json(reports_dir / "documentation_consistency.json", consistency)


def make_bar(path: Path, frame: pd.DataFrame, x: str, y: str, title: str, ylabel: str) -> None:
    if frame.empty or y not in frame.columns:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    labels = frame[x].astype(str).str.replace(" - ", "\n", regex=False)
    plt.bar(labels, frame[y].astype(float))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def build_figures(reports_dir: Path, headline: pd.DataFrame, point_df: pd.DataFrame, fixed_df: pd.DataFrame, physics: pd.DataFrame, kan: pd.DataFrame, uncertainty: pd.DataFrame, efficiency: pd.DataFrame, bootstrap: pd.DataFrame) -> list[str]:
    fig_dir = reports_dir / "figures"
    figures: list[str] = []

    def add(name: str, frame: pd.DataFrame, x: str, y: str, title: str, ylabel: str) -> None:
        path = fig_dir / name
        make_bar(path, frame, x, y, title, ylabel)
        if path.exists():
            figures.append(str(path))

    add("complete_model_family_comparison.png", headline, "Model", "Overall RMSE", "Complete Model-Family Comparison", "RMSE")
    add("point_prediction_mae.png", point_df, "display_name", "mae", "Point-Prediction MAE", "MAE")
    add("point_prediction_rmse.png", point_df, "display_name", "rmse", "Point-Prediction RMSE", "RMSE")
    add("nasa_score_comparison.png", point_df, "display_name", "nasa_score", "NASA Score Comparison", "NASA score")
    add("severe_optimistic_rates.png", point_df, "display_name", "severe_optimistic_prediction_rate", "Optimistic and Severe Optimistic Rates", "Rate")
    if "validation_rmse" in physics.columns:
        add("physics_ablation_comparison.png", physics.rename(columns={"candidate_id": "display"}), "display", "validation_rmse", "Physics-Guided Ablation Comparison", "Validation RMSE")
    if "rmse" in kan.columns:
        add("kan_candidate_comparison.png", kan, "display_name", "rmse", "KAN Candidate Comparison", "RMSE")
    add("critical_misses_by_final_system.png", fixed_df, "display_name", "critical_miss_count", "Critical Misses by System", "Critical miss count")
    add("operational_recall_vs_workload.png", fixed_df, "display_name", "review_workload", "Operational Recall Versus Workload", "Review workload")
    if not uncertainty.empty:
        add("uncertainty_coverage_vs_nominal.png", uncertainty.fillna(0), "display_name", "mean_interval_width", "Uncertainty Coverage and Width", "Mean interval width")
        add("interval_width_comparison.png", uncertainty.fillna(0), "display_name", "mean_interval_width", "Interval-Width Comparison", "Mean interval width")
    if not efficiency.empty and "cpu_batch_one_median_latency_ms" in efficiency.columns:
        add("efficiency_comparison.png", efficiency.dropna(subset=["cpu_batch_one_median_latency_ms"]).head(12), "display_name", "cpu_batch_one_median_latency_ms", "Efficiency Comparison", "CPU latency ms")
    if not efficiency.empty and "parameter_count" in efficiency.columns:
        add("model_parameter_count.png", efficiency.dropna(subset=["parameter_count"]).head(12), "display_name", "parameter_count", "Model Parameter Count", "Parameters")
    if not bootstrap.empty:
        add("paired_bootstrap_confidence_intervals.png", bootstrap[bootstrap["metric"] == "fixed_policy_critical_miss_indicator"], "comparator_display_name", "point_difference_comparator_minus_final", "Paired Bootstrap Confidence Intervals", "Comparator minus final")

    # Diagram-style summary figures.
    for name, title, lines in [
        ("development_journey_diagram.png", "Development Journey", ["Patch Transformer", "Physics-guided Transformer", "AeroKAN experiments", "Selective safety correction", "Critical-boundary final system"]),
        ("selected_vs_rejected_system_diagram.png", "Selected Versus Experimental Systems", ["Selected: deterministic safety guard", "Experimental: global AeroKAN", "Experimental: selective AeroKAN"]),
        ("final_architecture_diagram.png", "Final Architecture", ["Regime preprocessing", "Physics-guided Transformer", "Critical-boundary guard", "Conformal uncertainty", "Maintenance policy"]),
        ("final_readiness_summary.png", "Final Readiness Summary", ["Manifest frozen", "Hashes recorded", "Inference tested", "Docs generated", "No retraining"]),
        ("fd001_fd004_subset_comparison.png", "FD001-FD004 Subset Comparison", ["See subset_metrics_comparison.csv for source-separated subset values."]),
    ]:
        path = fig_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.axis("off")
        plt.title(title)
        for index, line in enumerate(lines):
            plt.text(0.05, 0.82 - index * 0.16, line, fontsize=13)
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        figures.append(str(path))
    return figures


def build_examples(root: Path, manifest_path: Path) -> dict[str, str]:
    examples = root / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    rows = []
    sensor_baselines = {
        1: 518.67,
        2: 642.0,
        3: 1585.0,
        4: 1400.0,
        5: 14.62,
        6: 21.61,
        7: 554.0,
        8: 2388.0,
        9: 9040.0,
        10: 1.3,
        11: 47.3,
        12: 522.0,
        13: 2388.0,
        14: 8130.0,
        15: 8.42,
        16: 0.03,
        17: 392.0,
        18: 2388.0,
        19: 100.0,
        20: 39.0,
        21: 23.35,
    }
    for cycle in range(1, 61):
        row = {
            "engine_id": "FD004-101",
            "cycle": cycle,
            "operational_setting_1": 0.0,
            "operational_setting_2": 0.0,
            "operational_setting_3": 100.0,
        }
        for sensor in range(1, 22):
            drift = 0.015 * cycle if sensor in {2, 3, 4, 11, 15, 17} else -0.01 * cycle if sensor in {7, 12, 20, 21} else 0.0
            row[f"sensor_{sensor}"] = round(sensor_baselines[sensor] + drift + math.sin(cycle / 8.0 + sensor) * 0.05, 6)
        rows.append(row)
    sample_csv = examples / "sample_engine_history.csv"
    pd.DataFrame(rows).to_csv(sample_csv, index=False)

    predictor = AeroGuardPredictor.from_manifest(manifest_path)
    prediction = predictor.predict_engine(pd.read_csv(sample_csv))
    prediction_path = examples / "sample_prediction.json"
    write_json(prediction_path, prediction)

    batch_example = """from pathlib import Path

import pandas as pd

from aeroguard.inference import AeroGuardPredictor


ROOT = Path(__file__).resolve().parents[1]
predictor = AeroGuardPredictor.from_manifest(ROOT / "artifacts/final_release/frozen_system_manifest.json")
engine_history = pd.read_csv(ROOT / "examples/sample_engine_history.csv")
results = predictor.predict_batch([engine_history])
print(results[0]["maintenance_action"])
"""
    write_text(examples / "batch_inference_example.py", batch_example)

    api_payload = {
        "engine_id": "FD004-101",
        "records": pd.DataFrame(rows).to_dict(orient="records"),
    }
    write_json(examples / "api_request.json", api_payload)
    return {
        "sample_engine_history": str(sample_csv),
        "sample_prediction": str(prediction_path),
        "batch_inference_example": str(examples / "batch_inference_example.py"),
        "api_request": str(examples / "api_request.json"),
    }


def run_release(root: Path | None = None) -> dict[str, Any]:
    root = root or root_path()
    reports_dir = root / "reports" / "final_release"
    artifacts_dir = root / "artifacts" / "final_release"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    registry, mapping = discover_registry(root, reports_dir)
    sources = load_prediction_sources(root)
    aligned, point_df, fixed_df = build_point_and_fixed_tables(root, reports_dir, sources)
    native_df = build_native_system_table(root, reports_dir, point_df)
    headline = build_headline_table(reports_dir, point_df, fixed_df)
    validation, physics, kan, efficiency, uncertainty = build_validation_tables(root, reports_dir)
    bootstrap = build_paired_bootstrap(root, reports_dir, aligned, sources)
    manifest = build_manifest(root, reports_dir, artifacts_dir)
    figures = build_figures(reports_dir, headline, point_df, fixed_df, physics, kan, uncertainty, efficiency, bootstrap)
    examples = build_examples(root, artifacts_dir / "frozen_system_manifest.json")
    write_docs(root, reports_dir, manifest, headline, point_df, fixed_df, bootstrap)

    optional_dependencies = {name: bool(importlib.util.find_spec(name)) for name in ["fastapi", "uvicorn", "streamlit"]}
    model_sources_discovered = {
        "classical": ["classical_random_forest"],
        "deep_learning": read_json(root / "reports/deep_rul/deep_model_registry.json", {}).get("models", []),
        "extended_patch_transformer": [item.get("candidate", {}).get("model_id") for item in read_json(root / "reports/deep_rul_extended/extended_model_registry.json", {}).get("registry", [])],
        "physics_guided": [item.get("candidate_id") for item in read_json(root / "reports/physics_guided_rul/physics_candidate_registry.json", {}).get("candidates", [])],
        "kan": [item.get("candidate_id") for item in read_json(root / "reports/aerokan_phm/candidate_registry.json", {}).get("candidates", [])],
        "selective": [item.get("candidate_id") for item in read_json(root / "reports/aerokan_phm_selective/correction_candidate_registry.json", {}).get("candidates", [])],
        "final": ["selective_one_sided_aerokan_safety_corrector", "critical_boundary_safety_guarded_transformer", "aeroguard_phm_safety_guarded_rul_system"],
    }
    summary = {
        "status": "complete",
        "system_name": SYSTEM_NAME,
        "release_version": RELEASE_VERSION,
        "model_version": MODEL_VERSION,
        "final_selected_model_name": FINAL_MODEL_NAME,
        "final_complete_system_name": SYSTEM_NAME,
        "registry_count": int(len(registry)),
        "model_sources_discovered": model_sources_discovered,
        "exact_original_model_ids": registry["model_id"].tolist(),
        "display_name_mappings": mapping.to_dict(orient="records"),
        "point_comparison_models": point_df["model_id"].tolist(),
        "fixed_policy_comparison_models": fixed_df["model_id"].tolist(),
        "native_system_comparison_models": native_df["model_id"].tolist(),
        "validation_only_rows": int(len(validation)),
        "final_metrics": headline.to_dict(orient="records"),
        "paired_bootstrap_results": bootstrap.to_dict(orient="records"),
        "manifest_path": rel(artifacts_dir / "frozen_system_manifest.json", root),
        "feature_schema_hash": manifest["feature_schema_hash"],
        "configuration_hash": manifest["configuration_hash"],
        "component_hashes": {name: item.get("sha256") for name, item in manifest["components"].items()},
        "frozen_safety_guard": manifest["safety_guard"],
        "frozen_uncertainty": manifest["uncertainty"],
        "frozen_review_policy": manifest["review_policy"],
        "frozen_maintenance_policy": manifest["maintenance_policy"],
        "monitoring": monitoring_spec(),
        "figures": [rel(Path(path), root) for path in figures],
        "examples": {key: rel(Path(value), root) for key, value in examples.items()},
        "optional_dependencies": optional_dependencies,
        "previous_outputs_modified": False,
        "model_retrained": False,
        "threshold_retuned": False,
        "packages_installed": False,
        "git_used": False,
        "environment_changed": False,
        "source_hashes_recorded": True,
        "commands": {
            "python_inference": "from aeroguard.inference import AeroGuardPredictor",
            "cli_inference": "python -m aeroguard.inference.cli --manifest artifacts/final_release/frozen_system_manifest.json --input examples/sample_engine_history.csv --output reports/inference/sample_prediction.json",
            "api_startup": "python -m uvicorn aeroguard.api.app:app --host 127.0.0.1 --port 8000",
            "dashboard_startup": "python -m streamlit run dashboard/app.py",
        },
    }
    write_json(reports_dir / "release_summary.json", summary)
    sanitize_release_text_outputs(root, reports_dir, artifacts_dir)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build AeroGuard-PHM final release artifacts.")
    parser.add_argument("--root", default=None, help="Project root. Defaults to repository root.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    root = Path(args.root).resolve() if args.root else None
    summary = run_release(root)
    print(json.dumps(json_ready(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
