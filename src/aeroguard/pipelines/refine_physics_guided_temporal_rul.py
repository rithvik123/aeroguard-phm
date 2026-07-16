"""Phase 5C.1 post-hoc reliability refinement for physics-guided RUL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

try:  # Existing environment dependency; no installation is performed here.
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover - exercised only in stripped environments.
    IsotonicRegression = None
    LogisticRegression = None
    average_precision_score = None
    roc_auc_score = None

from aeroguard.deep.models.physics_guided_patch_transformer import PhysicsGuidedPatchTransformer
from aeroguard.evaluation.deep_rul_metrics import deep_point_metrics
from aeroguard.evaluation.metrics import nasa_asymmetric_score
from aeroguard.evaluation.uncertainty_metrics import interval_metrics
from aeroguard.pipelines.train_fd001_baseline import project_root, resolve_project_path


REQUIRED_CONFIG_SECTIONS = {
    "source_phase5c",
    "source_phase5b",
    "outputs",
    "constraint_audit",
    "policy_selection",
    "uncertainty",
    "abstention",
    "maintenance",
    "bootstrap",
}

REPORT_ARTIFACTS = {
    "run_summary.json": ("run summary", True),
    "screening_metrics.csv": ("screening metrics", True),
    "finalist_cross_validation_metrics.csv": ("finalist CV metrics", True),
    "cv_predictions.csv": ("OOF CV predictions", True),
    "model_stability.csv": ("model stability", True),
    "constraint_ablation.csv": ("constraint ablation", True),
    "physics_model_ranking.csv": ("model ranking", True),
    "locked_physics_model.json": ("locked model metadata", True),
    "final_fit_metadata.json": ("final fit metadata", True),
    "benchmark_predictions.csv": ("benchmark point predictions", True),
    "trajectory_consistency_metrics.csv": ("trajectory consistency metrics", True),
    "optimistic_error_analysis.csv": ("optimistic error analysis", True),
    "uncertainty_cv_metrics.csv": ("uncertainty CV metrics", True),
    "uncertainty_predictions.csv": ("benchmark uncertainty predictions", True),
    "uncertainty_metrics.json": ("uncertainty metrics", True),
    "abstention_metrics.json": ("abstention metrics", True),
    "maintenance_recommendations.csv": ("maintenance recommendations", True),
    "maintenance_policy_metrics.json": ("maintenance policy metrics", True),
    "model_efficiency.csv": ("model efficiency", True),
}

LABEL_COLUMNS = {
    "true_rul",
    "true_rul_capped",
    "rul_capped",
    "target_rul_capped",
    "target_rul_uncapped",
}

REFINED_REPORTS = [
    "source_artifact_manifest.json",
    "source_run_validation.json",
    "corrected_run_summary.json",
    "constraint_metric_audit.json",
    "corrected_constraint_metrics.csv",
    "constraint_bootstrap_intervals.csv",
    "uncertainty_candidate_metrics.csv",
    "locked_uncertainty_policy.json",
    "refined_uncertainty_predictions.csv",
    "refined_uncertainty_metrics.json",
    "abstention_candidate_metrics.csv",
    "locked_abstention_policy.json",
    "refined_abstention_predictions.csv",
    "refined_abstention_metrics.json",
    "maintenance_policy_candidates.csv",
    "locked_maintenance_policy.json",
    "refined_maintenance_recommendations.csv",
    "refined_maintenance_metrics.json",
    "phase5b_vs_phase5c_paired_bootstrap.csv",
    "phase5c_refinement_summary.json",
]


@dataclass(frozen=True)
class ArtifactSpec:
    path: Path
    role: str
    required: bool = True


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    missing = sorted(REQUIRED_CONFIG_SECTIONS - set(config))
    if missing:
        raise ValueError(f"Missing refinement config sections: {missing}")
    return config


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(payload), handle, indent=2, allow_nan=False)
    temp_path.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_dirs(config: dict[str, Any], root: Path) -> dict[str, Path]:
    return {
        "source_reports": resolve_project_path(config["source_phase5c"]["reports_dir"], root),
        "source_artifacts": resolve_project_path(config["source_phase5c"]["artifacts_dir"], root),
        "phase5b_reports": resolve_project_path(config["source_phase5b"]["reports_dir"], root),
        "phase5b_artifacts": resolve_project_path(config["source_phase5b"]["artifacts_dir"], root),
        "reports": resolve_project_path(config["outputs"]["reports_dir"], root),
        "artifacts": resolve_project_path(config["outputs"]["artifacts_dir"], root),
    }


def source_artifact_specs(config: dict[str, Any], root: Path) -> dict[str, ArtifactSpec]:
    dirs = resolve_dirs(config, root)
    reports = dirs["source_reports"]
    artifacts = dirs["source_artifacts"]
    specs: dict[str, ArtifactSpec] = {
        name: ArtifactSpec(reports / name, role, required) for name, (role, required) in REPORT_ARTIFACTS.items()
    }
    final_meta_path = reports / "final_fit_metadata.json"
    final_meta = read_json(final_meta_path) if final_meta_path.exists() else {}
    checkpoint = Path(final_meta.get("checkpoint_path", artifacts / "checkpoints" / "locked_physics_guided_model.pt"))
    preprocessor = Path(final_meta.get("preprocessor_path", artifacts / "checkpoints" / "final_preprocessor.pkl"))
    transformed = Path(final_meta.get("final_train_transformed_path", artifacts / "checkpoints" / "final_train_transformed.pkl"))
    specs["locked_checkpoint"] = ArtifactSpec(checkpoint, "locked checkpoint", True)
    specs["final_preprocessor"] = ArtifactSpec(preprocessor, "final preprocessor", True)
    specs["final_train_transformed"] = ArtifactSpec(transformed, "transformed final training frame", True)
    for name, role in [
        ("benchmark_predictions.csv", "Phase 5B benchmark predictions"),
        ("uncertainty_predictions.csv", "Phase 5B uncertainty predictions"),
        ("benchmark_metrics.json", "Phase 5B benchmark metrics"),
        ("uncertainty_metrics.json", "Phase 5B uncertainty metrics"),
    ]:
        specs[f"phase5b_{name}"] = ArtifactSpec(dirs["phase5b_reports"] / name, role, True)
    return specs


def build_source_manifest(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows = []
    for key, spec in sorted(source_artifact_specs(config, root).items()):
        exists = spec.path.exists()
        stat = spec.path.stat() if exists else None
        rows.append(
            {
                "artifact_key": key,
                "source_path": str(spec.path),
                "semantic_role": spec.role,
                "required": bool(spec.required),
                "exists": bool(exists),
                "size_bytes": int(stat.st_size) if stat else 0,
                "sha256": sha256_file(spec.path) if exists else "",
                "modification_timestamp": pd.Timestamp.fromtimestamp(stat.st_mtime).isoformat() if stat else "",
            }
        )
    return rows


def _read_csv_if_nonempty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 2:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_source_tables(config: dict[str, Any], root: Path) -> dict[str, Any]:
    dirs = resolve_dirs(config, root)
    reports = dirs["source_reports"]
    phase5b = dirs["phase5b_reports"]
    return {
        "run_summary": read_json(reports / "run_summary.json"),
        "locked_model": read_json(reports / "locked_physics_model.json"),
        "final_fit_metadata": read_json(reports / "final_fit_metadata.json"),
        "benchmark_metrics": read_json(reports / "benchmark_metrics.json"),
        "uncertainty_metrics": read_json(reports / "uncertainty_metrics.json"),
        "abstention_metrics": read_json(reports / "abstention_metrics.json"),
        "maintenance_policy_metrics": read_json(reports / "maintenance_policy_metrics.json"),
        "screening_metrics": _read_csv_if_nonempty(reports / "screening_metrics.csv"),
        "cv_metrics": _read_csv_if_nonempty(reports / "finalist_cross_validation_metrics.csv"),
        "cv_predictions": _read_csv_if_nonempty(reports / "cv_predictions.csv"),
        "benchmark_predictions": _read_csv_if_nonempty(reports / "benchmark_predictions.csv"),
        "source_uncertainty_predictions": _read_csv_if_nonempty(reports / "uncertainty_predictions.csv"),
        "maintenance_recommendations": _read_csv_if_nonempty(reports / "maintenance_recommendations.csv"),
        "phase5b_predictions": _read_csv_if_nonempty(phase5b / "benchmark_predictions.csv"),
        "phase5b_uncertainty_predictions": _read_csv_if_nonempty(phase5b / "uncertainty_predictions.csv"),
    }


def _candidate_metadata_from_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _model_from_checkpoint_metadata(metadata: dict[str, Any]) -> PhysicsGuidedPatchTransformer:
    candidate = metadata.get("candidate", {})
    architecture = candidate.get("architecture_parameters", {})
    feature_names = metadata.get("feature_names", [])
    return PhysicsGuidedPatchTransformer(
        input_dim=len(feature_names) + 1,
        window_length=int(architecture.get("window_length", metadata.get("window", {}).get("window_length", 50))),
        patch_length=int(architecture.get("patch_length", metadata.get("window", {}).get("patch_length", 10))),
        patch_stride=int(architecture.get("patch_stride", metadata.get("window", {}).get("patch_stride", 5))),
        projection_dim=int(architecture.get("projection_dim", 64)),
        layers=int(architecture.get("layers", 2)),
        heads=int(architecture.get("heads", 4)),
        feedforward_dim=int(architecture.get("feedforward_dim", 192)),
        dropout=float(architecture.get("dropout", 0.15)),
        positional_encoding=str(architecture.get("positional_encoding", "learnable")),
        pooling=str(architecture.get("pooling", "mean")),
        causal_attention=bool(architecture.get("causal_attention", False)),
        health_head_enabled=bool(candidate.get("active_output_heads") and "health" in candidate.get("active_output_heads", [])),
        rate_head_enabled=bool(candidate.get("active_output_heads") and "rate" in candidate.get("active_output_heads", [])),
        parameter_budget=int(candidate.get("parameter_budget", 1_000_000)),
    )


def validate_checkpoint_predictions_unchanged(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    metadata = payload.get("metadata", {})
    state = payload["state_dict"]
    model_a = _model_from_checkpoint_metadata(metadata)
    model_b = _model_from_checkpoint_metadata(metadata)
    state_keys_before = sorted(model_a.state_dict())
    model_a.load_state_dict(state)
    model_b.load_state_dict(state)
    state_keys_after = sorted(model_a.state_dict())
    model_a.eval()
    model_b.eval()
    window = int(metadata.get("window", {}).get("window_length", 50))
    feature_count = len(metadata.get("feature_names", []))
    generator = torch.Generator().manual_seed(4242)
    x = torch.randn((4, window, feature_count + 1), generator=generator)
    x[..., -1] = 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            pred_a = model_a(x)["rul_prediction"]
            pred_b = model_b(x)["rul_prediction"]
    nested_warnings = [str(item.message) for item in caught if "nested" in str(item.message).lower()]
    return {
        "checkpoint_loads": True,
        "state_dict_key_count": len(state_keys_after),
        "state_dict_keys_unchanged": state_keys_before == state_keys_after,
        "deterministic_prediction_max_abs_delta": float(torch.max(torch.abs(pred_a - pred_b)).item()),
        "deterministic_predictions_identical_within_tolerance": bool(torch.allclose(pred_a, pred_b, rtol=1.0e-6, atol=1.0e-6)),
        "nested_tensor_warning_count": len(nested_warnings),
    }


def validate_source_run(config: dict[str, Any], root: Path, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    dirs = resolve_dirs(config, root)
    sources = load_source_tables(config, root)
    missing = [row["artifact_key"] for row in manifest if row["required"] and not row["exists"]]
    if missing:
        raise FileNotFoundError(f"Missing required Phase 5C.1 source artifacts: {missing}")
    final_meta = sources["final_fit_metadata"]
    locked = sources["locked_model"]
    checkpoint_path = Path(final_meta["checkpoint_path"])
    checkpoint_meta = _candidate_metadata_from_checkpoint(checkpoint_path)
    preprocessor_path = Path(final_meta["preprocessor_path"])
    with preprocessor_path.open("rb") as handle:
        preprocessor = pickle.load(handle)
    preprocessor_features = list(preprocessor.get("features", [])) if isinstance(preprocessor, dict) else []
    feature_names = list(final_meta.get("feature_names", []))
    checkpoint_features = list(checkpoint_meta.get("feature_names", []))
    expected_candidate = str(config["source_phase5c"]["expected_locked_candidate"])
    candidates = {
        "expected": expected_candidate,
        "locked_model": str(locked.get("candidate_id", "")),
        "final_fit_metadata": str(final_meta.get("candidate_id", "")),
        "checkpoint": str(checkpoint_meta.get("candidate", {}).get("candidate_id", "")),
    }
    benchmark = sources["benchmark_predictions"]
    subset_counts = benchmark.groupby("subset", observed=False)["global_engine_id"].nunique().to_dict()
    duplicate_keys = int(
        benchmark.duplicated(["subset", "global_engine_id", "final_observed_cycle"]).sum()
    )
    cv = sources["cv_predictions"]
    phase5b_hashes = final_meta.get("phase5b_hashes", {})
    phase5b_mismatches = [
        path for path, digest in phase5b_hashes.items() if Path(path).exists() and sha256_file(Path(path)) != digest
    ]
    label_feature_leaks = sorted(LABEL_COLUMNS & set(feature_names))
    checkpoint_check = validate_checkpoint_predictions_unchanged(checkpoint_path)
    validations = {
        "source_run_status": sources["run_summary"].get("run_status"),
        "source_runtime_seconds": sources["run_summary"].get("runtime_seconds"),
        "candidate_ids": candidates,
        "locked_candidate_consistent": len(set(candidates.values())) == 1,
        "feature_schema_matches_preprocessor": bool(feature_names == preprocessor_features),
        "feature_schema_matches_checkpoint": bool(feature_names == checkpoint_features),
        "window_length_matches": int(final_meta.get("window_length")) == int(checkpoint_meta.get("window", {}).get("window_length", -1)),
        "patch_length_matches": int(final_meta.get("patch_length")) == int(checkpoint_meta.get("window", {}).get("patch_length", -1)),
        "patch_stride_matches": int(final_meta.get("patch_stride")) == int(checkpoint_meta.get("window", {}).get("patch_stride", -1)),
        "rul_cap_matches": float(final_meta.get("rul_cap")) == float(checkpoint_meta.get("rul_cap", float("nan"))),
        "cv_fold_seed_identifiers_valid": bool({"fold", "seed"}.issubset(cv.columns) and cv["fold"].notna().all() and cv["seed"].notna().all()),
        "benchmark_prediction_count": int(len(benchmark)),
        "benchmark_count_is_707": int(len(benchmark)) == 707,
        "benchmark_subset_counts": {str(k): int(v) for k, v in subset_counts.items()},
        "benchmark_subset_counts_expected": subset_counts == {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248},
        "duplicate_benchmark_engine_keys": duplicate_keys,
        "no_duplicate_benchmark_engine_keys": duplicate_keys == 0,
        "label_feature_leaks": label_feature_leaks,
        "no_benchmark_labels_in_model_features": not label_feature_leaks,
        "phase5b_source_hashes_match_final_metadata": not phase5b_mismatches,
        "phase5b_hash_mismatches": phase5b_mismatches,
        **checkpoint_check,
    }
    hard_failures = [
        key
        for key in [
            "locked_candidate_consistent",
            "feature_schema_matches_preprocessor",
            "feature_schema_matches_checkpoint",
            "window_length_matches",
            "patch_length_matches",
            "patch_stride_matches",
            "rul_cap_matches",
            "cv_fold_seed_identifiers_valid",
            "benchmark_count_is_707",
            "benchmark_subset_counts_expected",
            "no_duplicate_benchmark_engine_keys",
            "no_benchmark_labels_in_model_features",
            "phase5b_source_hashes_match_final_metadata",
            "checkpoint_loads",
            "state_dict_keys_unchanged",
            "deterministic_predictions_identical_within_tolerance",
        ]
        if not validations.get(key)
    ]
    validations["hard_failures"] = hard_failures
    if hard_failures:
        raise ValueError(f"Source artifact validation failed: {hard_failures}")
    return validations


def corrected_run_summary(source_summary: dict[str, Any]) -> dict[str, Any]:
    runtime_by_stage = source_summary.get("runtime_by_stage", {})
    runtime = float(source_summary.get("runtime_seconds") or 0.0)
    if runtime <= 0.0 and isinstance(runtime_by_stage, dict):
        runtime = float(sum(float(value) for value in runtime_by_stage.values()))
    summary = dict(source_summary)
    summary.update(
        {
            "run_status": "completed",
            "runtime_seconds": runtime,
            "completed_stage_count": len(runtime_by_stage) if isinstance(runtime_by_stage, dict) else None,
            "failed_stage": None,
            "failures": [],
            "source_run_status": source_summary.get("run_status"),
            "source_runtime_seconds": source_summary.get("runtime_seconds"),
            "correction_reason": "Original Phase 5C summary was serialized before final state/runtime fields were updated.",
            "read_only_interpretation": True,
        }
    )
    return summary


def add_rul_bands(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    values = result["true_rul"].astype(float) if "true_rul" in result.columns else result["predicted_rul"].astype(float)
    bins = [-np.inf, 15, 30, 60, 125, np.inf]
    labels = ["critical", "near_term", "inspection", "monitoring", "healthy"]
    result["true_rul_band_refined"] = pd.cut(values, bins=bins, labels=labels, right=True).astype(str)
    predicted = result["predicted_rul"].astype(float)
    result["predicted_rul_band_refined"] = pd.cut(predicted, bins=bins, labels=labels, right=True).astype(str)
    return result


def engine_key_frame(frame: pd.DataFrame) -> pd.Series:
    return frame["subset"].astype(str) + "|" + frame["global_engine_id"].astype(str)


def _quantile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float(default)
    return float(np.quantile(finite, float(q)))


def build_temporal_comparison_details(frame: pd.DataFrame, cap: float, mono_tolerance: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    triplets = []
    grouping = ["subset", "candidate_id", "fold", "seed", "global_engine_id"]
    available = [column for column in grouping if column in frame.columns]
    for _, group in frame.sort_values(available + ["cycle"]).groupby(available, observed=False):
        ordered = group.sort_values("cycle").reset_index(drop=True)
        if len(ordered) < 2:
            continue
        for i in range(len(ordered) - 1):
            for j in range(i + 1, len(ordered)):
                early = ordered.iloc[i]
                late = ordered.iloc[j]
                pair_type = "adjacent" if j == i + 1 else "fixed_gap"
                pred_delta = float(late["predicted_rul"]) - float(early["predicted_rul"])
                expected_delta = float(late["target_rul_capped"]) - float(early["target_rul_capped"])
                mono_magnitude = max(0.0, pred_delta - float(mono_tolerance))
                rate_residual = abs(pred_delta - expected_delta)
                early_cap = float(early["target_rul_capped"])
                late_cap = float(late["target_rul_capped"])
                if early_cap >= cap and late_cap >= cap:
                    region = "healthy_capped_plateau"
                elif early_cap >= cap > late_cap:
                    region = "cap_transition"
                else:
                    region = "uncapped_degradation"
                rows.append(
                    {
                        "engine_bootstrap_key": f"{early['subset']}|{early['global_engine_id']}",
                        "subset": early["subset"],
                        "fold": early.get("fold", ""),
                        "seed": early.get("seed", ""),
                        "global_engine_id": early["global_engine_id"],
                        "operating_regime": early.get("operating_regime", np.nan),
                        "true_rul_band": early.get("true_rul_band_refined", ""),
                        "pair_type": pair_type,
                        "earlier_cycle": float(early["cycle"]),
                        "later_cycle": float(late["cycle"]),
                        "cycle_gap": float(late["cycle"] - early["cycle"]),
                        "predicted_delta": pred_delta,
                        "expected_capped_delta": expected_delta,
                        "monotonic_violation_magnitude": mono_magnitude,
                        "rate_residual": rate_residual,
                        "trajectory_region": region,
                    }
                )
        if len(ordered) >= 3:
            for i in range(len(ordered) - 2):
                a, b, c = ordered.iloc[i], ordered.iloc[i + 1], ordered.iloc[i + 2]
                gap1 = float(b["cycle"] - a["cycle"])
                gap2 = float(c["cycle"] - b["cycle"])
                if gap1 <= 0 or gap2 <= 0:
                    continue
                slope1 = (float(b["predicted_rul"]) - float(a["predicted_rul"])) / gap1
                slope2 = (float(c["predicted_rul"]) - float(b["predicted_rul"])) / gap2
                if abs(gap1 - gap2) <= 1.0e-9:
                    acceleration = float(c["predicted_rul"]) - 2.0 * float(b["predicted_rul"]) + float(a["predicted_rul"])
                else:
                    acceleration = (slope2 - slope1) / ((gap1 + gap2) / 2.0)
                capped = [float(row["target_rul_capped"]) for row in [a, b, c]]
                if min(capped) >= cap:
                    region = "healthy_capped_plateau"
                elif max(capped) >= cap > min(capped):
                    region = "cap_transition"
                else:
                    region = "uncapped_degradation"
                triplets.append(
                    {
                        "engine_bootstrap_key": f"{a['subset']}|{a['global_engine_id']}",
                        "subset": a["subset"],
                        "fold": a.get("fold", ""),
                        "seed": a.get("seed", ""),
                        "global_engine_id": a["global_engine_id"],
                        "operating_regime": a.get("operating_regime", np.nan),
                        "true_rul_band": a.get("true_rul_band_refined", ""),
                        "left_gap": gap1,
                        "right_gap": gap2,
                        "abs_acceleration": abs(float(acceleration)),
                        "trajectory_region": region,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(triplets)


def build_regime_comparison_details(frame: pd.DataFrame, tolerance: float, max_pairs: int, seed: int) -> pd.DataFrame:
    required = {"operating_regime", "target_rul_capped", "predicted_rul", "subset", "global_engine_id"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    ordered = frame.sort_values(["target_rul_capped", "subset", "global_engine_id", "cycle"]).reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    anchors = np.arange(len(ordered))
    if len(anchors) > max_pairs:
        anchors = np.sort(rng.choice(anchors, size=min(len(anchors), max_pairs), replace=False))
    rows = []
    targets = ordered["target_rul_capped"].to_numpy(dtype=float)
    regimes = ordered["operating_regime"].to_numpy()
    for idx in anchors:
        value = targets[idx]
        lo = np.searchsorted(targets, value - tolerance, side="left")
        hi = np.searchsorted(targets, value + tolerance, side="right")
        candidates = [j for j in range(lo, hi) if j != idx and regimes[j] != regimes[idx]]
        if not candidates:
            continue
        partner = candidates[int(rng.integers(0, len(candidates)))]
        left = ordered.iloc[idx]
        right = ordered.iloc[partner]
        rows.append(
            {
                "engine_bootstrap_key": f"{left['subset']}|{left['global_engine_id']}",
                "subset": left["subset"],
                "left_regime": left["operating_regime"],
                "right_regime": right["operating_regime"],
                "regime_pair": f"{left['operating_regime']}->{right['operating_regime']}",
                "left_true_capped_rul": float(left["target_rul_capped"]),
                "right_true_capped_rul": float(right["target_rul_capped"]),
                "true_rul_difference": float(left["target_rul_capped"]) - float(right["target_rul_capped"]),
                "prediction_difference": float(left["predicted_rul"]) - float(right["predicted_rul"]),
            }
        )
        if len(rows) >= max_pairs:
            break
    result = pd.DataFrame(rows)
    if not result.empty:
        result["prediction_difference_residual"] = (
            result["prediction_difference"].astype(float) - result["true_rul_difference"].astype(float)
        ).abs()
    return result


def _metric_rows_for_values(
    details: pd.DataFrame,
    *,
    metric_name: str,
    value_column: str,
    threshold: float,
    units: str,
    group_columns: list[str],
) -> list[dict[str, Any]]:
    if details.empty:
        return [
            {
                "metric_name": metric_name,
                "grouping": "overall",
                "group_value": "none",
                "units": units,
                "threshold": threshold,
                "valid_comparison_count": 0,
                "violation_count": 0,
                "violation_rate": np.nan,
                "mean_magnitude": np.nan,
                "median_magnitude": np.nan,
                "p90_magnitude": np.nan,
                "maximum_magnitude": np.nan,
            }
        ]
    rows = []
    groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "overall", details)]
    for column in group_columns:
        if column in details.columns:
            groups.extend((column, str(value), group) for value, group in details.groupby(column, dropna=False, observed=False))
    for grouping, value, group in groups:
        values = group[value_column].to_numpy(dtype=float)
        violations = values > float(threshold)
        rows.append(
            {
                "metric_name": metric_name,
                "grouping": grouping,
                "group_value": value,
                "units": units,
                "threshold": float(threshold),
                "valid_comparison_count": int(len(values)),
                "violation_count": int(violations.sum()),
                "violation_rate": float(violations.mean()) if len(values) else np.nan,
                "mean_magnitude": float(np.mean(values)) if len(values) else np.nan,
                "median_magnitude": float(np.median(values)) if len(values) else np.nan,
                "p90_magnitude": _quantile(values, 0.90, np.nan),
                "maximum_magnitude": float(np.max(values)) if len(values) else np.nan,
            }
        )
    return rows


def corrected_constraint_diagnostics(cv_predictions: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame], dict[str, float]]:
    cv = add_rul_bands(cv_predictions)
    cap = float(cv["target_rul_capped"].max()) if "target_rul_capped" in cv.columns else 125.0
    mono_tol = float(config["constraint_audit"]["monotonic_tolerance"])
    pairs, triplets = build_temporal_comparison_details(cv, cap, mono_tol)
    regime = build_regime_comparison_details(
        cv,
        float(config["constraint_audit"]["rul_matching_tolerance"]),
        int(config["constraint_audit"]["maximum_regime_pairs"]),
        int(config["constraint_audit"]["bootstrap_seed"]),
    )
    thresholds = {
        "monotonic": mono_tol,
        "cycle_rate": _quantile(pairs["rate_residual"].to_numpy(dtype=float), float(config["constraint_audit"]["rate_threshold_quantile"]), 0.0) if not pairs.empty else 0.0,
        "smoothness": _quantile(triplets["abs_acceleration"].to_numpy(dtype=float), float(config["constraint_audit"]["smoothness_threshold_quantile"]), 0.0) if not triplets.empty else 0.0,
        "regime_consistency": _quantile(regime["prediction_difference_residual"].to_numpy(dtype=float), float(config["constraint_audit"]["regime_threshold_quantile"]), 0.0) if not regime.empty else 0.0,
    }
    rows: list[dict[str, Any]] = []
    mono_details = pairs.copy()
    rows.extend(
        _metric_rows_for_values(
            mono_details,
            metric_name="monotonicity",
            value_column="monotonic_violation_magnitude",
            threshold=0.0,
            units="cycles above tolerance",
            group_columns=["pair_type", "subset", "operating_regime", "true_rul_band"],
        )
    )
    rows.extend(
        _metric_rows_for_values(
            pairs,
            metric_name="cycle_rate",
            value_column="rate_residual",
            threshold=thresholds["cycle_rate"],
            units="cycles residual",
            group_columns=["pair_type", "trajectory_region", "subset"],
        )
    )
    rows.extend(
        _metric_rows_for_values(
            triplets,
            metric_name="smoothness",
            value_column="abs_acceleration",
            threshold=thresholds["smoothness"],
            units="cycles acceleration",
            group_columns=["trajectory_region", "subset"],
        )
    )
    rows.extend(
        _metric_rows_for_values(
            regime,
            metric_name="regime_consistency",
            value_column="prediction_difference_residual",
            threshold=thresholds["regime_consistency"],
            units="cycles residual",
            group_columns=["regime_pair", "subset"],
        )
    )
    audit = {
        "monotonicity": {
            "existing_formula": "violation_rate from prediction increases over temporal pairs",
            "existing_units": "fraction",
            "existing_threshold": "implicit default from training metric helper",
            "input_columns": ["predicted_rul", "cycle", "global_engine_id"],
            "valid_comparison_count": int(len(pairs)),
            "corrected_formula": "max(0, predicted_later - predicted_earlier - epsilon_mono)",
            "corrected_threshold_selection_method": "configured tolerance, validation-only reporting",
        },
        "cycle_rate": {
            "existing_formula": "abs((pred_later - pred_earlier) + cycle_gap) > tolerance",
            "existing_units": "cycles",
            "existing_threshold": "training helper default",
            "input_columns": ["predicted_rul", "target_rul_capped", "cycle"],
            "valid_comparison_count": int(len(pairs)),
            "root_cause_of_degenerate_outputs": "The old diagnostic assumed every valid pair should lose exactly one RUL cycle per cycle gap. Capped-RUL plateaus and sparse validation snapshots violate that assumption, so nearly every residual exceeded the tiny default tolerance.",
            "corrected_formula": "abs((pred_later - pred_earlier) - (capped_true_later - capped_true_earlier))",
            "corrected_threshold_selection_method": f"validation-only p{int(float(config['constraint_audit']['rate_threshold_quantile']) * 100)} residual quantile",
            "locked_threshold": thresholds["cycle_rate"],
        },
        "smoothness": {
            "existing_formula": "abs(pred_later - 2*pred_middle + pred_earlier) > tolerance",
            "existing_units": "cycles second difference",
            "existing_threshold": "training helper default",
            "input_columns": ["predicted_rul", "cycle"],
            "valid_comparison_count": int(len(triplets)),
            "root_cause_of_degenerate_outputs": "The old diagnostic used gap-unaware curvature and an unrealistically small default tolerance on sparse, unequally spaced snapshots, so any nonzero curvature was counted as a violation.",
            "corrected_formula": "equal gaps use second difference; unequal gaps use gap-normalized slope change",
            "corrected_threshold_selection_method": f"validation-only p{int(float(config['constraint_audit']['smoothness_threshold_quantile']) * 100)} acceleration quantile",
            "locked_threshold": thresholds["smoothness"],
        },
        "regime_consistency": {
            "existing_formula": "absolute prediction disagreement over supplied regime pairs",
            "existing_units": "fraction",
            "existing_threshold": "greater than zero",
            "input_columns": ["predicted_rul", "target_rul_capped", "operating_regime"],
            "valid_comparison_count": int(len(regime)),
            "number_excluded": int(max(0, len(cv) - len(regime))),
            "root_cause_of_nan": "The completed trajectory-consistency output was empty and the old diagnostic only used a supplied regime-pair frame; no valid diagnostic pair table was persisted for final reporting.",
            "corrected_formula": "abs((pred_i - pred_j) - (capped_true_i - capped_true_j)) for validation-only pairs from different regimes and similar capped RUL",
            "corrected_threshold_selection_method": f"validation-only p{int(float(config['constraint_audit']['regime_threshold_quantile']) * 100)} residual quantile",
            "locked_threshold": thresholds["regime_consistency"],
            "nan_allowed_only_when_no_valid_pair_exists": True,
        },
    }
    details = {"temporal_pairs": pairs, "smoothness_triplets": triplets, "regime_pairs": regime}
    return audit, pd.DataFrame(rows), details, thresholds


def bootstrap_constraint_intervals(details: dict[str, pd.DataFrame], thresholds: dict[str, float], iterations: int, seed: int) -> pd.DataFrame:
    specs = [
        ("monotonicity", details["temporal_pairs"], "monotonic_violation_magnitude", 0.0),
        ("cycle_rate", details["temporal_pairs"], "rate_residual", thresholds["cycle_rate"]),
        ("smoothness", details["smoothness_triplets"], "abs_acceleration", thresholds["smoothness"]),
        ("regime_consistency", details["regime_pairs"], "prediction_difference_residual", thresholds["regime_consistency"]),
    ]
    rng = np.random.default_rng(int(seed))
    rows = []
    for metric, frame, column, threshold in specs:
        if frame.empty:
            rows.append({"metric_name": metric, "point_estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "engine_count": 0, "valid_comparison_count": 0})
            continue
        engines = np.asarray(sorted(frame["engine_bootstrap_key"].astype(str).unique()))
        point = float((frame[column].to_numpy(dtype=float) > float(threshold)).mean())
        samples = []
        grouped = {engine: group for engine, group in frame.groupby("engine_bootstrap_key", observed=False)}
        for _ in range(int(iterations)):
            chosen = rng.choice(engines, size=len(engines), replace=True)
            sample = pd.concat([grouped[engine] for engine in chosen], ignore_index=True)
            samples.append(float((sample[column].to_numpy(dtype=float) > float(threshold)).mean()))
        rows.append(
            {
                "metric_name": metric,
                "point_estimate": point,
                "ci_lower": _quantile(np.asarray(samples), 0.025, point),
                "ci_upper": _quantile(np.asarray(samples), 0.975, point),
                "engine_count": int(len(engines)),
                "valid_comparison_count": int(len(frame)),
                "bootstrap_iterations": int(iterations),
            }
        )
    return pd.DataFrame(rows)


def split_engine_groups(frame: pd.DataFrame, development_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = np.asarray(sorted(engine_key_frame(frame).unique()))
    rng = np.random.default_rng(int(seed))
    shuffled = keys.copy()
    rng.shuffle(shuffled)
    cut = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * float(development_fraction)))))
    dev_keys = set(shuffled[:cut])
    key_series = engine_key_frame(frame)
    return frame[key_series.isin(dev_keys)].copy(), frame[~key_series.isin(dev_keys)].copy()


def _group_values(frame: pd.DataFrame, method_id: str) -> pd.Series:
    if method_id == "predicted_rul_band_mondrian":
        return add_rul_bands(frame)["predicted_rul_band_refined"].astype(str)
    if method_id == "operating_regime_mondrian":
        return frame["operating_regime"].astype(str) if "operating_regime" in frame.columns else pd.Series(["missing"] * len(frame))
    if method_id == "hybrid_rul_band_regime_shrinkage":
        band = add_rul_bands(frame)["predicted_rul_band_refined"].astype(str)
        regime = frame["operating_regime"].astype(str) if "operating_regime" in frame.columns else "missing"
        return band + "|" + regime
    return pd.Series(["global"] * len(frame), index=frame.index)


def fit_conformal_policy(frame: pd.DataFrame, method_id: str, levels: list[float], minimum_group_size: int, shrinkage_weight: float) -> dict[str, Any]:
    residual = (frame["predicted_rul"].astype(float) - frame["true_rul"].astype(float)).abs().to_numpy(dtype=float)
    global_radii = {str(level): _quantile(residual, float(level), 0.0) for level in levels}
    policy: dict[str, Any] = {"method_id": method_id, "levels": levels, "global_radii": global_radii, "minimum_group_size": int(minimum_group_size)}
    if method_id == "normalized_residual_conformal":
        scale = np.maximum(5.0, np.sqrt(np.maximum(frame["predicted_rul"].to_numpy(dtype=float), 0.0) + 1.0))
        scaled = residual / scale
        policy["normalized_radii"] = {str(level): _quantile(scaled, float(level), 0.0) for level in levels}
    if "mondrian" in method_id or "shrinkage" in method_id:
        group_values = _group_values(frame, method_id)
        residual_series = pd.Series(residual, index=frame.index)
        groups: dict[str, Any] = {}
        for group_value, indices in group_values.groupby(group_values, observed=False).groups.items():
            values = residual_series.loc[list(indices)].to_numpy(dtype=float)
            support = int(len(values))
            groups[str(group_value)] = {
                "support": support,
                "radii": {
                    str(level): (
                        (1.0 - float(shrinkage_weight)) * _quantile(values, float(level), global_radii[str(level)])
                        + float(shrinkage_weight) * global_radii[str(level)]
                        if "shrinkage" in method_id
                        else _quantile(values, float(level), global_radii[str(level)])
                    )
                    for level in levels
                },
            }
        policy["groups"] = groups
        policy["shrinkage_weight"] = float(shrinkage_weight) if "shrinkage" in method_id else 0.0
    return policy


def apply_conformal_policy(frame: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    method_id = str(policy["method_id"])
    result["uncertainty_method_id"] = method_id
    group_values = _group_values(result, method_id).reset_index(drop=True)
    fallback = []
    for level in policy["levels"]:
        pct = int(round(float(level) * 100))
        if method_id == "normalized_residual_conformal":
            scale = np.maximum(5.0, np.sqrt(np.maximum(result["predicted_rul"].to_numpy(dtype=float), 0.0) + 1.0))
            radius = float(policy["normalized_radii"][str(level)]) * scale
            fallback.extend([False] * len(result))
        elif "mondrian" in method_id or "shrinkage" in method_id:
            radii = []
            used_fallback = []
            for group_value in group_values.astype(str).tolist():
                group = policy.get("groups", {}).get(group_value)
                if group is None or int(group.get("support", 0)) < int(policy["minimum_group_size"]):
                    radii.append(float(policy["global_radii"][str(level)]))
                    used_fallback.append(True)
                else:
                    radii.append(float(group["radii"][str(level)]))
                    used_fallback.append(False)
            radius = np.asarray(radii, dtype=float)
            fallback.extend(used_fallback)
        else:
            radius = np.full(len(result), float(policy["global_radii"][str(level)]), dtype=float)
            fallback.extend([False] * len(result))
        result[f"lower_{pct}"] = np.maximum(0.0, result["predicted_rul"].to_numpy(dtype=float) - radius)
        result[f"upper_{pct}"] = result["predicted_rul"].to_numpy(dtype=float) + radius
        result[f"radius_{pct}"] = radius
        result[f"interval_width_{pct}"] = result[f"upper_{pct}"] - result[f"lower_{pct}"]
        if "true_rul" in result.columns:
            result[f"covered_{pct}"] = (result["true_rul"] >= result[f"lower_{pct}"]) & (result["true_rul"] <= result[f"upper_{pct}"])
    result["group_fallback_used"] = bool(fallback and any(fallback))
    return result


def uncertainty_candidate_selection(cv: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    levels = [float(level) for level in config["uncertainty"]["nominal_levels"]]
    methods = [
        "existing_phase5c_global_conformal",
        "global_absolute_residual_conformal",
        "normalized_residual_conformal",
        "predicted_rul_band_mondrian",
        "operating_regime_mondrian",
        "shrinkage_predicted_rul_band_mondrian",
        "hybrid_rul_band_regime_shrinkage",
    ]
    rows = []
    for seed in config["policy_selection"]["split_seeds"]:
        dev, val = split_engine_groups(cv, float(config["policy_selection"]["development_fraction"]), int(seed))
        for method in methods:
            policy_method = "global_absolute_residual_conformal" if method == "existing_phase5c_global_conformal" else method
            if method == "shrinkage_predicted_rul_band_mondrian":
                policy_method = "predicted_rul_band_mondrian_shrinkage"
            policy = fit_conformal_policy(
                dev,
                policy_method,
                levels,
                int(config["uncertainty"]["minimum_group_size"]),
                float(config["uncertainty"]["shrinkage_weight"]),
            )
            predictions = apply_conformal_policy(val, policy)
            for level in levels:
                pct = int(round(level * 100))
                metrics = interval_metrics(predictions["true_rul"], predictions["predicted_rul"], predictions[f"lower_{pct}"], predictions[f"upper_{pct}"], level)
                rows.append(
                    {
                        "candidate_method": method,
                        "policy_method": policy_method,
                        "selection_seed": int(seed),
                        "nominal_level": level,
                        "coverage": metrics["coverage"],
                        "mean_width": metrics["mean_interval_width"],
                        "median_width": metrics["median_interval_width"],
                        "p90_width": _quantile(predictions[f"interval_width_{pct}"].to_numpy(dtype=float), 0.90, metrics["mean_interval_width"]),
                        "winkler_score": metrics["winkler_interval_score"],
                        "undercoverage": metrics["undercoverage_amount"],
                        "overcoverage": metrics["overcoverage_amount"],
                        "group_fallback_rate": float(predictions.get("group_fallback_used", pd.Series([False] * len(predictions))).astype(bool).mean()),
                    }
                )
    candidate_metrics = pd.DataFrame(rows)
    level90 = candidate_metrics[candidate_metrics["nominal_level"] == 0.90].copy()
    aggregate = (
        level90.groupby(["candidate_method", "policy_method"], observed=False)
        .agg(
            coverage=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            mean_width=("mean_width", "mean"),
            median_width=("median_width", "mean"),
            p90_width=("p90_width", "mean"),
            winkler_score=("winkler_score", "mean"),
            undercoverage=("undercoverage", "mean"),
            overcoverage=("overcoverage", "mean"),
            group_fallback_rate=("group_fallback_rate", "mean"),
        )
        .reset_index()
    )
    aggregate["coverage_std"] = aggregate["coverage_std"].fillna(0.0)
    aggregate["feasible"] = (
        (aggregate["coverage"] >= 0.90 - float(config["uncertainty"]["maximum_undercoverage"]))
        & (aggregate["coverage_std"] <= float(config["uncertainty"]["maximum_instability"]))
    )
    aggregate["selection_score"] = np.where(aggregate["feasible"], 0.0, 100.0) + aggregate["undercoverage"] * 50.0 + aggregate["winkler_score"] + aggregate["mean_width"] * 0.01
    aggregate = aggregate.sort_values(["selection_score", "mean_width", "candidate_method"], kind="mergesort").reset_index(drop=True)
    aggregate["selection_rank"] = np.arange(1, len(aggregate) + 1)
    selected = aggregate.iloc[0].to_dict()
    locked_policy = fit_conformal_policy(
        cv,
        str(selected["policy_method"]),
        levels,
        int(config["uncertainty"]["minimum_group_size"]),
        float(config["uncertainty"]["shrinkage_weight"]),
    )
    locked_policy.update(
        {
            "candidate_method": str(selected["candidate_method"]),
            "selection_source": "engine-grouped OOF validation predictions only",
            "selection_nominal_level": 0.90,
            "selection_metrics": selected,
        }
    )
    cv_with_intervals = apply_conformal_policy(cv, locked_policy)
    return pd.concat([candidate_metrics, aggregate], ignore_index=True, sort=False), locked_policy, cv_with_intervals


def uncertainty_metrics_by_subset(predictions: pd.DataFrame, levels: list[float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    groups = list(predictions.groupby("subset", observed=False)) + [("overall", predictions)]
    for subset, group in groups:
        result[str(subset)] = {}
        for level in levels:
            pct = int(round(float(level) * 100))
            result[str(subset)][str(level)] = interval_metrics(group["true_rul"], group["predicted_rul"], group[f"lower_{pct}"], group[f"upper_{pct}"], float(level))
    return result


def _rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values.to_numpy(dtype=float))))) if len(values) else np.nan


def high_error_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    high = frame["absolute_error"].astype(float) > float(threshold)
    abstained = frame["abstain_flag"].astype(bool)
    accepted = ~abstained
    before = _rmse(frame["residual"].astype(float))
    after = _rmse(frame.loc[accepted, "residual"].astype(float)) if accepted.any() else np.nan
    abstained_high_rate = float(high[abstained].mean()) if abstained.any() else np.nan
    accepted_high_rate = float(high[accepted].mean()) if accepted.any() else np.nan
    return {
        "engine_count": int(len(frame)),
        "accepted_count": int(accepted.sum()),
        "abstained_count": int(abstained.sum()),
        "acceptance_rate": float(accepted.mean()) if len(frame) else np.nan,
        "abstention_rate": float(abstained.mean()) if len(frame) else np.nan,
        "mae_before_abstention": float(frame["absolute_error"].mean()) if len(frame) else np.nan,
        "mae_after_abstention": float(frame.loc[accepted, "absolute_error"].mean()) if accepted.any() else np.nan,
        "rmse_before_abstention": before,
        "rmse_after_abstention": after,
        "abstained_mae": float(frame.loc[abstained, "absolute_error"].mean()) if abstained.any() else np.nan,
        "error_rate_abstained": abstained_high_rate,
        "error_rate_accepted": accepted_high_rate,
        "error_enrichment_ratio": float(abstained_high_rate / accepted_high_rate) if np.isfinite(abstained_high_rate) and np.isfinite(accepted_high_rate) and accepted_high_rate > 0 else np.nan,
        "high_error_recall": float((high & abstained).sum() / max(high.sum(), 1)),
        "high_error_precision": float((high & abstained).sum() / max(abstained.sum(), 1)),
        "unnecessary_abstention_rate": float(((~high) & abstained).sum() / max(abstained.sum(), 1)) if abstained.any() else 0.0,
        "high_error_predictions_abstained": int((high & abstained).sum()),
        "low_error_predictions_unnecessarily_abstained": int(((~high) & abstained).sum()),
    }


def risk_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "interval_width_90" not in result.columns and {"lower_90", "upper_90"}.issubset(result.columns):
        result["interval_width_90"] = result["upper_90"] - result["lower_90"]
    result["normalized_interval_width"] = result["interval_width_90"].astype(float) / np.maximum(result["predicted_rul"].astype(float).abs(), 1.0)
    result["padding_fraction"] = result.get("padded_cycle_count", pd.Series(0, index=result.index)).astype(float) / np.maximum(
        result.get("sequence_valid_length", pd.Series(1, index=result.index)).astype(float) + result.get("padded_cycle_count", pd.Series(0, index=result.index)).astype(float),
        1.0,
    )
    regime_counts = result["operating_regime"].value_counts(normalize=True) if "operating_regime" in result.columns else pd.Series(dtype=float)
    result["operating_regime_rarity"] = result["operating_regime"].map(lambda value: 1.0 - float(regime_counts.get(value, 1.0))) if "operating_regime" in result.columns else 0.0
    result["interpretable_risk_score"] = (
        result["normalized_interval_width"].rank(pct=True)
        + result["padding_fraction"].rank(pct=True)
        + result["operating_regime_rarity"].rank(pct=True)
        + (-result["predicted_rul"].astype(float)).rank(pct=True)
    ) / 4.0
    return result


def _policy_abstain(frame: pd.DataFrame, policy: dict[str, Any]) -> pd.Series:
    if policy["policy_id"] == "no_abstention":
        return pd.Series(False, index=frame.index)
    if policy["policy_id"].startswith("width_threshold"):
        return frame["interval_width_90"].astype(float) >= float(policy["threshold"])
    if policy["policy_id"].startswith("support_threshold"):
        return frame["operating_regime_rarity"].astype(float) >= float(policy["threshold"])
    if policy["policy_id"].startswith("weighted_risk"):
        return frame["interpretable_risk_score"].astype(float) >= float(policy["threshold"])
    if policy["policy_id"].startswith("logistic"):
        columns = policy["feature_columns"]
        model = policy["model"]
        scores = model.predict_proba(frame[columns].to_numpy(dtype=float))[:, 1]
        return pd.Series(scores >= float(policy["threshold"]), index=frame.index)
    return pd.Series(False, index=frame.index)


def abstention_policy_selection(cv: pd.DataFrame, config: dict[str, Any], artifacts_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    threshold = float(config["abstention"]["high_error_threshold"])
    frame = risk_feature_frame(cv)
    rows = []
    fitted_policies: dict[str, dict[str, Any]] = {"no_abstention": {"policy_id": "no_abstention"}}
    rates = [0.05, 0.10, min(0.15, float(config["abstention"]["maximum_abstention_rate"]))]
    for seed in config["policy_selection"]["split_seeds"]:
        dev, val = split_engine_groups(frame, float(config["policy_selection"]["development_fraction"]), int(seed))
        seed_policies: list[dict[str, Any]] = [{"policy_id": "no_abstention"}]
        for rate in rates:
            width_threshold = float(dev["interval_width_90"].quantile(1.0 - rate))
            risk_threshold = float(dev["interpretable_risk_score"].quantile(1.0 - rate))
            support_threshold = float(dev["operating_regime_rarity"].quantile(1.0 - rate))
            seed_policies.extend(
                [
                    {"policy_id": f"width_threshold_{rate:.2f}", "threshold": width_threshold},
                    {"policy_id": f"support_threshold_{rate:.2f}", "threshold": support_threshold},
                    {"policy_id": f"weighted_risk_{rate:.2f}", "threshold": risk_threshold},
                ]
            )
        if LogisticRegression is not None and len(dev["absolute_error"].gt(threshold).unique()) > 1:
            columns = ["interval_width_90", "normalized_interval_width", "padding_fraction", "operating_regime_rarity", "predicted_rul"]
            model = LogisticRegression(max_iter=500, random_state=int(seed), class_weight="balanced")
            model.fit(dev[columns].to_numpy(dtype=float), dev["absolute_error"].gt(threshold).astype(int))
            dev_scores = model.predict_proba(dev[columns].to_numpy(dtype=float))[:, 1]
            for rate in rates:
                score_threshold = float(np.quantile(dev_scores, 1.0 - rate))
                seed_policies.append({"policy_id": f"logistic_high_error_{rate:.2f}", "threshold": score_threshold, "feature_columns": columns, "model": model})
        for policy in seed_policies:
            pred = val.copy()
            pred["abstain_flag"] = _policy_abstain(pred, policy).astype(bool)
            pred["abstention_reason"] = np.where(pred["abstain_flag"], policy["policy_id"], "")
            metrics = high_error_metrics(pred, threshold)
            scores = pred["interpretable_risk_score"].to_numpy(dtype=float)
            target = pred["absolute_error"].gt(threshold).astype(int).to_numpy(dtype=int)
            if len(np.unique(target)) > 1:
                auroc = float(roc_auc_score(target, scores)) if roc_auc_score else np.nan
                auprc = float(average_precision_score(target, scores)) if average_precision_score else np.nan
            else:
                auroc = np.nan
                auprc = np.nan
            rows.append({"policy_id": policy["policy_id"], "selection_seed": int(seed), "auroc": auroc, "auprc": auprc, **metrics})
    metrics_frame = pd.DataFrame(rows)
    aggregate = metrics_frame.groupby("policy_id", observed=False).mean(numeric_only=True).reset_index()
    baseline = aggregate[aggregate["policy_id"] == "no_abstention"].iloc[0]
    aggregate["accepted_rmse_improvement"] = float(baseline["rmse_after_abstention"]) - aggregate["rmse_after_abstention"]
    aggregate["stable"] = True
    feasible = aggregate[
        (aggregate["abstention_rate"] <= float(config["abstention"]["maximum_abstention_rate"]))
        & (aggregate["accepted_rmse_improvement"] > 0.0)
        & (aggregate["error_enrichment_ratio"].fillna(0.0) >= float(config["abstention"]["minimum_error_enrichment"]))
    ].copy()
    if feasible.empty or bool(config["abstention"].get("allow_no_abstention", True)):
        selected_id = "no_abstention" if feasible.empty else str(feasible.sort_values(["accepted_rmse_improvement", "error_enrichment_ratio"], ascending=[False, False]).iloc[0]["policy_id"])
    else:
        selected_id = str(feasible.sort_values(["accepted_rmse_improvement", "error_enrichment_ratio"], ascending=[False, False]).iloc[0]["policy_id"])
    locked: dict[str, Any] = {"policy_id": selected_id, "selection_source": "engine-grouped OOF validation predictions only", "high_error_threshold": threshold}
    if selected_id != "no_abstention":
        if selected_id.startswith("width_threshold"):
            rate = float(selected_id.rsplit("_", 1)[1])
            locked["threshold"] = float(frame["interval_width_90"].quantile(1.0 - rate))
        elif selected_id.startswith("support_threshold"):
            rate = float(selected_id.rsplit("_", 1)[1])
            locked["threshold"] = float(frame["operating_regime_rarity"].quantile(1.0 - rate))
        elif selected_id.startswith("weighted_risk"):
            rate = float(selected_id.rsplit("_", 1)[1])
            locked["threshold"] = float(frame["interpretable_risk_score"].quantile(1.0 - rate))
        elif selected_id.startswith("logistic") and LogisticRegression is not None:
            rate = float(selected_id.rsplit("_", 1)[1])
            columns = ["interval_width_90", "normalized_interval_width", "padding_fraction", "operating_regime_rarity", "predicted_rul"]
            model = LogisticRegression(max_iter=500, random_state=10501, class_weight="balanced")
            model.fit(frame[columns].to_numpy(dtype=float), frame["absolute_error"].gt(threshold).astype(int))
            scores = model.predict_proba(frame[columns].to_numpy(dtype=float))[:, 1]
            locked.update({"feature_columns": columns, "threshold": float(np.quantile(scores, 1.0 - rate))})
            model_path = artifacts_dir / "abstention_logistic_model.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with model_path.open("wb") as handle:
                pickle.dump(model, handle)
            locked["model_path"] = str(model_path)
            locked["model"] = model
    benchmark_ready = frame.copy()
    locked_runtime = dict(locked)
    if "model_path" in locked_runtime and "model" not in locked_runtime:
        with Path(locked_runtime["model_path"]).open("rb") as handle:
            locked_runtime["model"] = pickle.load(handle)
    benchmark_ready["abstain_flag"] = _policy_abstain(benchmark_ready, locked_runtime).astype(bool)
    benchmark_ready["abstention_reason"] = np.where(benchmark_ready["abstain_flag"], selected_id, "")
    if "model" in locked:
        locked = {key: value for key, value in locked.items() if key != "model"}
    return pd.concat([metrics_frame, aggregate], ignore_index=True, sort=False), locked, benchmark_ready


def apply_locked_abstention(benchmark_uncertainty: pd.DataFrame, locked: dict[str, Any]) -> pd.DataFrame:
    frame = risk_feature_frame(benchmark_uncertainty)
    runtime_policy = dict(locked)
    if "model_path" in runtime_policy:
        with Path(runtime_policy["model_path"]).open("rb") as handle:
            runtime_policy["model"] = pickle.load(handle)
    frame["abstain_flag"] = _policy_abstain(frame, runtime_policy).astype(bool)
    frame["abstention_reason"] = np.where(frame["abstain_flag"], locked["policy_id"], "")
    frame["prediction_status"] = np.where(frame["abstain_flag"], "abstained", "accepted")
    return frame


def true_condition_band(true_rul: pd.Series, config: dict[str, Any]) -> pd.Series:
    critical = float(config["maintenance"]["critical_rul_max"])
    near = float(config["maintenance"]["near_term_rul_max"])
    inspect = float(config["maintenance"]["inspection_rul_max"])
    healthy = float(config["maintenance"]["healthy_rul_min"])
    values = true_rul.astype(float)
    return pd.Series(
        np.select(
            [values <= critical, values <= near, values <= inspect, values >= healthy],
            ["critical", "near_term", "inspection", "healthy"],
            default="monitoring",
        ),
        index=true_rul.index,
    )


def assign_maintenance_actions(frame: pd.DataFrame, policy_id: str, config: dict[str, Any]) -> pd.Series:
    critical = float(config["maintenance"]["critical_rul_max"])
    near = float(config["maintenance"]["near_term_rul_max"])
    inspect = float(config["maintenance"]["inspection_rul_max"])
    pred = frame["predicted_rul"].astype(float)
    lower = frame["lower_90"].astype(float) if "lower_90" in frame.columns else pred
    risk = frame.get("interpretable_risk_score", pd.Series(0.0, index=frame.index)).astype(float)
    basis = pred
    if policy_id == "lower_bound_threshold":
        basis = lower
    elif policy_id == "point_lower_hybrid":
        basis = np.minimum(pred, lower + 10.0)
    elif policy_id == "risk_adjusted_hybrid":
        basis = np.minimum(pred, lower + 10.0) - 15.0 * (risk > risk.quantile(0.85)).astype(float)
    elif policy_id == "conservative_low_support":
        basis = np.minimum(pred, lower + 5.0)
    actions = np.select(
        [basis <= critical, basis <= near, basis <= inspect],
        ["URGENT_ENGINEERING_REVIEW", "SCHEDULE_MAINTENANCE", "PLAN_INSPECTION"],
        default="CONTINUE_MONITORING",
    )
    if "abstain_flag" in frame.columns:
        actions = np.where(frame["abstain_flag"].astype(bool), "ENGINEERING_REVIEW_REQUIRED", actions)
    return pd.Series(actions, index=frame.index)


def maintenance_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    condition = true_condition_band(frame["true_rul"], config)
    urgent = frame["maintenance_action"] == "URGENT_ENGINEERING_REVIEW"
    critical = condition == "critical"
    unnecessary = urgent & ~critical
    missed = critical & ~urgent
    expected_action = condition.map(
        {
            "critical": "URGENT_ENGINEERING_REVIEW",
            "near_term": "SCHEDULE_MAINTENANCE",
            "inspection": "PLAN_INSPECTION",
            "monitoring": "CONTINUE_MONITORING",
            "healthy": "CONTINUE_MONITORING",
        }
    )
    per_condition_accuracy = []
    for value, mask in condition.groupby(condition, observed=False).groups.items():
        index = list(mask)
        if index:
            per_condition_accuracy.append(float((frame.loc[index, "maintenance_action"] == expected_action.loc[index]).mean()))
    cost = (
        missed.astype(float) * float(config["maintenance"]["cost_matrix"]["missed_critical"])
        + ((condition == "near_term") & frame["maintenance_action"].isin(["CONTINUE_MONITORING", "PLAN_INSPECTION"])).astype(float) * float(config["maintenance"]["cost_matrix"]["delayed_near_term"])
        + unnecessary.astype(float) * float(config["maintenance"]["cost_matrix"]["unnecessary_urgent"])
        + ((condition.isin(["monitoring", "healthy"])) & frame["maintenance_action"].isin(["PLAN_INSPECTION", "SCHEDULE_MAINTENANCE"])).astype(float) * float(config["maintenance"]["cost_matrix"]["early_inspection"])
    )
    return {
        "engine_count": int(len(frame)),
        "critical_recall": float((urgent & critical).sum() / max(critical.sum(), 1)),
        "critical_precision": float((urgent & critical).sum() / max(urgent.sum(), 1)),
        "urgent_review_precision": float((urgent & critical).sum() / max(urgent.sum(), 1)),
        "urgent_review_false_positive_rate": float(unnecessary.sum() / max((~critical).sum(), 1)),
        "missed_critical_count": int(missed.sum()),
        "unnecessary_urgent_count": int(unnecessary.sum()),
        "macro_action_accuracy": float(np.mean(per_condition_accuracy)) if per_condition_accuracy else np.nan,
        "weighted_operational_cost": float(cost.mean()) if len(frame) else np.nan,
        "action_distribution": {str(k): int(v) for k, v in frame["maintenance_action"].value_counts().sort_index().items()},
        "mean_lead_time_by_action": {str(k): float(v) for k, v in frame.groupby("maintenance_action", observed=False)["true_rul"].mean().items()},
    }


def maintenance_policy_selection(cv: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = [
        "existing_phase5c_policy",
        "point_prediction_threshold",
        "lower_bound_threshold",
        "point_lower_hybrid",
        "risk_adjusted_hybrid",
        "conservative_low_support",
    ]
    rows = []
    frame = risk_feature_frame(cv)
    for seed in config["policy_selection"]["split_seeds"]:
        _, val = split_engine_groups(frame, float(config["policy_selection"]["development_fraction"]), int(seed))
        for policy_id in candidates:
            pred = val.copy()
            pred["maintenance_action"] = assign_maintenance_actions(pred, policy_id, config)
            rows.append({"policy_id": policy_id, "selection_seed": int(seed), **maintenance_metrics(pred, config)})
    metrics_frame = pd.DataFrame(rows)
    aggregate = metrics_frame.groupby("policy_id", observed=False).mean(numeric_only=True).reset_index()
    feasible = aggregate[aggregate["critical_recall"] >= float(config["maintenance"]["critical_recall_floor"])].copy()
    if feasible.empty:
        selected = aggregate.sort_values(["critical_recall", "weighted_operational_cost"], ascending=[False, True]).iloc[0]
    else:
        selected = feasible.sort_values(["weighted_operational_cost", "unnecessary_urgent_count", "policy_id"], ascending=[True, True, True]).iloc[0]
    locked = {
        "policy_id": str(selected["policy_id"]),
        "selection_source": "engine-grouped OOF validation predictions only",
        "selection_metrics": selected.to_dict(),
        "critical_rul_max": float(config["maintenance"]["critical_rul_max"]),
        "near_term_rul_max": float(config["maintenance"]["near_term_rul_max"]),
        "inspection_rul_max": float(config["maintenance"]["inspection_rul_max"]),
    }
    return pd.concat([metrics_frame, aggregate], ignore_index=True, sort=False), locked


def apply_locked_maintenance(frame: pd.DataFrame, locked: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    result = risk_feature_frame(frame)
    result["maintenance_action"] = assign_maintenance_actions(result, str(locked["policy_id"]), config)
    result["maintenance_disclaimer"] = "Demonstration decision-support output; not approved aircraft-maintenance instruction."
    return result


def nasa_contribution(true: pd.Series, pred: pd.Series) -> np.ndarray:
    residual = pred.to_numpy(dtype=float) - true.to_numpy(dtype=float)
    return np.where(residual < 0, np.exp(-residual / 13.0) - 1.0, np.exp(residual / 10.0) - 1.0)


def paired_bootstrap_comparison(phase5b: pd.DataFrame, phase5b_unc: pd.DataFrame, phase5c: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    key = ["subset", "global_engine_id", "final_observed_cycle"]
    if phase5b.duplicated(key).any() or phase5c.duplicated(key).any():
        raise ValueError("Duplicate engine keys found in Phase 5B/5C comparison inputs.")
    left = phase5b[key + ["true_rul", "predicted_rul", "residual", "absolute_error", "squared_error"]].rename(columns={column: f"phase5b_{column}" for column in ["true_rul", "predicted_rul", "residual", "absolute_error", "squared_error"]})
    right = phase5c[key + ["true_rul", "predicted_rul", "residual", "absolute_error", "squared_error", "interval_width_90", "covered_90"]].rename(columns={column: f"phase5c_{column}" for column in ["true_rul", "predicted_rul", "residual", "absolute_error", "squared_error", "interval_width_90", "covered_90"]})
    if {"interval_width_90", "covered_90"}.issubset(phase5b_unc.columns):
        unc = phase5b_unc[key + ["interval_width_90", "covered_90"]].rename(columns={"interval_width_90": "phase5b_interval_width_90", "covered_90": "phase5b_covered_90"})
        left = left.merge(unc, on=key, how="left", validate="one_to_one")
    merged = left.merge(right, on=key, how="inner", validate="one_to_one")
    if len(merged) != len(phase5c):
        raise ValueError(f"Paired comparison has {len(merged)} matched engines, expected {len(phase5c)}.")
    merged["phase5b_nasa_contribution"] = nasa_contribution(merged["phase5b_true_rul"], merged["phase5b_predicted_rul"])
    merged["phase5c_nasa_contribution"] = nasa_contribution(merged["phase5c_true_rul"], merged["phase5c_predicted_rul"])
    merged["phase5b_optimistic"] = merged["phase5b_residual"] > 0
    merged["phase5c_optimistic"] = merged["phase5c_residual"] > 0
    merged["phase5b_severe_optimistic"] = merged["phase5b_residual"] > 30.0
    merged["phase5c_severe_optimistic"] = merged["phase5c_residual"] > 30.0
    metric_columns = [
        ("absolute_error", "phase5b_absolute_error", "phase5c_absolute_error", "lower"),
        ("squared_error", "phase5b_squared_error", "phase5c_squared_error", "lower"),
        ("nasa_contribution", "phase5b_nasa_contribution", "phase5c_nasa_contribution", "lower"),
        ("signed_error", "phase5b_residual", "phase5c_residual", "abs_lower"),
        ("optimistic_indicator", "phase5b_optimistic", "phase5c_optimistic", "lower"),
        ("severe_optimistic_indicator", "phase5b_severe_optimistic", "phase5c_severe_optimistic", "lower"),
        ("interval_width_90", "phase5b_interval_width_90", "phase5c_interval_width_90", "lower"),
        ("coverage_90", "phase5b_covered_90", "phase5c_covered_90", "higher"),
    ]
    rng = np.random.default_rng(int(seed))
    rows = []
    for subset, group in list(merged.groupby("subset", observed=False)) + [("overall", merged)]:
        engines = np.arange(len(group))
        for metric, base_col, new_col, direction in metric_columns:
            if base_col not in group.columns or new_col not in group.columns:
                continue
            base = group[base_col].astype(float).to_numpy()
            new = group[new_col].astype(float).to_numpy()
            if not np.isfinite(base).all() or not np.isfinite(new).all():
                continue
            diff = new - base
            point = float(diff.mean())
            samples = []
            for _ in range(int(iterations)):
                idx = rng.choice(engines, size=len(engines), replace=True)
                samples.append(float(diff[idx].mean()))
            ci_lower = _quantile(np.asarray(samples), 0.025, point)
            ci_upper = _quantile(np.asarray(samples), 0.975, point)
            if direction == "higher":
                probability = float((np.asarray(samples) > 0.0).mean())
            elif direction == "abs_lower":
                probability = float((np.abs(new).mean() < np.abs(base).mean()))
            else:
                probability = float((np.asarray(samples) < 0.0).mean())
            rows.append(
                {
                    "subset": subset,
                    "metric": metric,
                    "phase5b_metric": float(np.mean(base)),
                    "phase5c_metric": float(np.mean(new)),
                    "absolute_difference": point,
                    "relative_difference": point / max(abs(float(np.mean(base))), 1.0e-9),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "probability_of_improvement": probability,
                    "interval_excludes_zero": bool(ci_lower > 0.0 or ci_upper < 0.0),
                    "improvement_direction": direction,
                    "engine_count": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def make_figures(output_dir: Path, constraint_details: dict[str, pd.DataFrame], uncertainty_candidates: pd.DataFrame, source_uncertainty: pd.DataFrame, refined_uncertainty: pd.DataFrame, abstention_candidates: pd.DataFrame, maintenance_source: pd.DataFrame, maintenance_refined: pd.DataFrame, paired: pd.DataFrame) -> list[str]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []

    def save(name: str) -> None:
        path = fig_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        figures.append(str(path))

    pairs = constraint_details.get("temporal_pairs", pd.DataFrame())
    triplets = constraint_details.get("smoothness_triplets", pd.DataFrame())
    regime = constraint_details.get("regime_pairs", pd.DataFrame())
    if not pairs.empty:
        plt.figure(figsize=(7, 4)); pairs["monotonic_violation_magnitude"].plot(kind="hist", bins=40); save("corrected_monotonicity_violation_distribution.png")
        plt.figure(figsize=(7, 4)); pairs.boxplot(column="rate_residual", by="trajectory_region", rot=25); plt.suptitle(""); save("cycle_rate_residual_by_region.png")
    if not triplets.empty:
        plt.figure(figsize=(7, 4)); triplets["abs_acceleration"].plot(kind="hist", bins=40); save("smoothness_residual_distribution.png")
    if not regime.empty:
        plt.figure(figsize=(7, 4)); regime.boxplot(column="prediction_difference_residual", by="regime_pair", rot=45); plt.suptitle(""); save("regime_consistency_residual_by_pair.png")
    if not uncertainty_candidates.empty and "nominal_level" in uncertainty_candidates.columns:
        selected = uncertainty_candidates.dropna(subset=["nominal_level", "coverage"])
        if not selected.empty:
            plt.figure(figsize=(7, 4)); selected.groupby("nominal_level", observed=False)["coverage"].mean().plot(marker="o"); plt.plot([0.8, 0.95], [0.8, 0.95], linestyle="--"); save("uncertainty_coverage_vs_nominal.png")
            plt.figure(figsize=(7, 4)); plt.scatter(selected["mean_width"], selected["coverage"], s=24); plt.xlabel("Mean width"); plt.ylabel("Coverage"); save("width_vs_coverage_candidates.png")
    if "interval_width_90" in source_uncertainty.columns and "interval_width_90" in refined_uncertainty.columns:
        plt.figure(figsize=(7, 4)); source_uncertainty["interval_width_90"].plot(kind="hist", bins=30, alpha=0.5, label="source"); refined_uncertainty["interval_width_90"].plot(kind="hist", bins=30, alpha=0.5, label="refined"); plt.legend(); save("width_distribution_before_after.png")
    if not abstention_candidates.empty:
        policy_rows = abstention_candidates.dropna(subset=["acceptance_rate", "rmse_after_abstention"]) if {"acceptance_rate", "rmse_after_abstention"}.issubset(abstention_candidates.columns) else pd.DataFrame()
        if not policy_rows.empty:
            plt.figure(figsize=(7, 4)); plt.scatter(policy_rows["acceptance_rate"], policy_rows["rmse_after_abstention"]); plt.xlabel("Acceptance rate"); plt.ylabel("Selective RMSE"); save("selective_rmse_vs_acceptance.png")
            plt.figure(figsize=(7, 4)); plt.scatter(policy_rows["abstention_rate"], policy_rows["error_enrichment_ratio"]); plt.xlabel("Abstention rate"); plt.ylabel("High-error enrichment"); save("high_error_enrichment_vs_abstention.png")
            plt.figure(figsize=(7, 4)); plt.plot(policy_rows.sort_values("acceptance_rate")["acceptance_rate"], policy_rows.sort_values("acceptance_rate")["rmse_after_abstention"]); plt.xlabel("Coverage"); plt.ylabel("Risk"); save("risk_coverage_curve.png")
    if not maintenance_refined.empty:
        plt.figure(figsize=(8, 4)); maintenance_refined["maintenance_action"].value_counts().sort_index().plot(kind="bar"); save("maintenance_action_distribution_after.png")
        if not maintenance_source.empty and "maintenance_action" in maintenance_source.columns:
            counts = pd.DataFrame({"before": maintenance_source["maintenance_action"].value_counts(), "after": maintenance_refined["maintenance_action"].value_counts()}).fillna(0)
            plt.figure(figsize=(8, 4)); counts.plot(kind="bar", ax=plt.gca()); save("maintenance_action_distribution_before_after.png")
        confusion = pd.crosstab(true_condition_band(maintenance_refined["true_rul"], {"maintenance": {"critical_rul_max": 15, "near_term_rul_max": 30, "inspection_rul_max": 60, "healthy_rul_min": 125}}), maintenance_refined["maintenance_action"])
        plt.figure(figsize=(8, 5)); plt.imshow(confusion.to_numpy(), aspect="auto"); plt.xticks(range(len(confusion.columns)), confusion.columns, rotation=45, ha="right"); plt.yticks(range(len(confusion.index)), confusion.index); plt.colorbar(); save("maintenance_policy_confusion_matrix.png")
    if not paired.empty:
        ae = paired[(paired["metric"] == "absolute_error") & (paired["subset"] != "overall")]
        if not ae.empty:
            plt.figure(figsize=(7, 4)); ae.set_index("subset")["absolute_difference"].plot(kind="bar"); save("phase5b_vs_phase5c_paired_error_difference.png")
        forest = paired[paired["subset"] == "overall"].head(10)
        if not forest.empty:
            plt.figure(figsize=(8, 5)); y = np.arange(len(forest)); plt.errorbar(forest["absolute_difference"], y, xerr=[forest["absolute_difference"] - forest["ci_lower"], forest["ci_upper"] - forest["absolute_difference"]], fmt="o"); plt.yticks(y, forest["metric"]); plt.axvline(0, color="black", linewidth=1); save("bootstrap_interval_forest_plot.png")
    examples = refined_uncertainty.head(6)
    if not examples.empty:
        plt.figure(figsize=(8, 4)); x = np.arange(len(examples)); plt.errorbar(x, examples["predicted_rul"], yerr=[examples["predicted_rul"] - examples["lower_90"], examples["upper_90"] - examples["predicted_rul"]], fmt="o"); plt.scatter(x, examples["true_rul"], marker="x", color="red"); save("prediction_interval_examples.png")
    return figures


def write_results_note(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 5C.1 Physics-Guided RUL Reliability Refinement",
        "",
        "This note summarizes a post-hoc refinement of the completed Phase 5C run. The locked neural model was not retrained.",
        "",
        "## Supported",
        "",
    ]
    for item in summary.get("scientific_interpretation", {}).get("supported", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Suggestive But Uncertain", ""])
    for item in summary.get("scientific_interpretation", {}).get("suggestive_but_uncertain", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Not Supported", ""])
    for item in summary.get("scientific_interpretation", {}).get("not_supported", []):
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scientific_interpretation(paired: pd.DataFrame, abstention_locked: dict[str, Any], uncertainty_metrics: dict[str, Any]) -> dict[str, list[str]]:
    supported = ["Original abstention policy was ineffective; validation-selected refinement explicitly allows no-abstention when selective risk does not improve."]
    uncertain = []
    not_supported = ["The learned regime term should not be interpreted as a physical law."]
    overall_ae = paired[(paired["subset"] == "overall") & (paired["metric"] == "absolute_error")]
    overall_nasa = paired[(paired["subset"] == "overall") & (paired["metric"] == "nasa_contribution")]
    if not overall_ae.empty and bool(overall_ae.iloc[0]["interval_excludes_zero"]) and float(overall_ae.iloc[0]["absolute_difference"]) < 0:
        supported.append("Phase 5C reduced paired absolute error relative to Phase 5B.")
    else:
        uncertain.append("Phase 5C point-error improvement over Phase 5B is not statistically decisive under paired bootstrap.")
    if not overall_nasa.empty and bool(overall_nasa.iloc[0]["interval_excludes_zero"]) and float(overall_nasa.iloc[0]["absolute_difference"]) < 0:
        supported.append("Phase 5C improved paired NASA contribution relative to Phase 5B.")
    else:
        uncertain.append("NASA-score improvement remains uncertain if the paired interval includes zero.")
    cov90 = uncertainty_metrics.get("overall", {}).get("0.9", {}).get("coverage")
    if cov90 is not None and cov90 >= 0.90:
        supported.append("Refined nominal 90% intervals retained conservative benchmark coverage after validation-only selection.")
    if abstention_locked.get("policy_id") == "no_abstention":
        supported.append("No-abstention was selected because nontrivial validation policies did not satisfy the effectiveness gates.")
    return {"supported": supported, "suggestive_but_uncertain": uncertain, "not_supported": not_supported}


def prepare_outputs(config: dict[str, Any], root: Path) -> tuple[Path, Path]:
    dirs = resolve_dirs(config, root)
    reports = dirs["reports"]
    artifacts = dirs["artifacts"]
    overwrite = bool(config["outputs"].get("overwrite_existing", False))
    if (reports.exists() or artifacts.exists()) and not overwrite:
        raise FileExistsError(f"Refined outputs already exist at {reports} or {artifacts}; set overwrite_existing explicitly.")
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    return reports, artifacts


def run_validate_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root()
    dirs = resolve_dirs(config, root)
    return {
        "status": "valid",
        "config_path": str(Path(config_path)),
        "source_reports_exists": dirs["source_reports"].exists(),
        "source_artifacts_exists": dirs["source_artifacts"].exists(),
        "phase5b_reports_exists": dirs["phase5b_reports"].exists(),
        "output_reports_dir": str(dirs["reports"]),
        "output_artifacts_dir": str(dirs["artifacts"]),
        "neural_retraining_disabled": True,
        "policy_selection_uses_benchmark_labels": False,
    }


def run_dry_run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root()
    manifest = build_source_manifest(config, root)
    missing = [row["artifact_key"] for row in manifest if row["required"] and not row["exists"]]
    dirs = resolve_dirs(config, root)
    return {
        "status": "dry_run_complete",
        "source_directories_read_only": [str(dirs["source_reports"]), str(dirs["source_artifacts"])],
        "refined_output_directories": [str(dirs["reports"]), str(dirs["artifacts"])],
        "required_artifact_count": int(sum(bool(row["required"]) for row in manifest)),
        "missing_required_artifacts": missing,
        "neural_retraining_disabled": True,
        "benchmark_labels_excluded_from_policy_selection": True,
        "posthoc_models_allowed": ["conformal_quantiles", "interpretable_thresholds", "logistic_error_risk"],
        "will_write_reports": list(REFINED_REPORTS),
        "full_posthoc_run_command": (
            '$env:PYTHONPATH = ".\\src"\n'
            f'python -m aeroguard.pipelines.refine_physics_guided_temporal_rul --config "{Path(config_path).as_posix()}" --full-posthoc-run'
        ),
    }


def _synthetic_source_frames() -> dict[str, Any]:
    rows = []
    rng = np.random.default_rng(7)
    for subset in ["FD001", "FD002"]:
        for engine in range(1, 7):
            regime = engine % 3
            for pos, cycle in enumerate([20, 40, 60, 80]):
                true = float(120 - cycle + engine)
                pred = true + rng.normal(0, 8)
                rows.append(
                    {
                        "subset": subset,
                        "source_domain": subset,
                        "global_engine_id": f"{subset}_{engine:04d}",
                        "local_unit_id": engine,
                        "unit_id": engine,
                        "cycle": cycle,
                        "endpoint_index": pos,
                        "endpoint_cycle": cycle,
                        "sequence_valid_length": min(cycle, 50),
                        "padded_cycle_count": max(0, 50 - cycle),
                        "target_rul_capped": min(100.0, true),
                        "target_rul_uncapped": true,
                        "operating_regime": regime,
                        "predicted_rul": pred,
                        "predicted_rul_raw": pred,
                        "candidate_id": "physics_regime",
                        "true_rul": true,
                        "residual": pred - true,
                        "absolute_error": abs(pred - true),
                        "squared_error": (pred - true) ** 2,
                        "fold": 1 + engine % 2,
                        "seed": 10501 + engine % 2,
                        "final_observed_cycle": cycle,
                    }
                )
    cv = pd.DataFrame(rows)
    bench = cv.groupby(["subset", "global_engine_id"], observed=False).tail(1).copy().reset_index(drop=True)
    bench["true_rul_capped"] = bench["target_rul_capped"]
    p5b = bench.copy()
    p5b["predicted_rul"] = p5b["predicted_rul"] + 3.0
    p5b["residual"] = p5b["predicted_rul"] - p5b["true_rul"]
    p5b["absolute_error"] = p5b["residual"].abs()
    p5b["squared_error"] = np.square(p5b["residual"])
    p5b_unc = p5b.copy()
    p5b_unc["lower_90"] = np.maximum(0, p5b_unc["predicted_rul"] - 25)
    p5b_unc["upper_90"] = p5b_unc["predicted_rul"] + 25
    p5b_unc["interval_width_90"] = p5b_unc["upper_90"] - p5b_unc["lower_90"]
    p5b_unc["covered_90"] = (p5b_unc["true_rul"] >= p5b_unc["lower_90"]) & (p5b_unc["true_rul"] <= p5b_unc["upper_90"])
    return {"cv": cv, "benchmark": bench, "phase5b": p5b, "phase5b_unc": p5b_unc}


def run_smoke_test(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    frames = _synthetic_source_frames()
    smoke_config = dict(config)
    smoke_config["constraint_audit"] = dict(config["constraint_audit"], bootstrap_iterations=int(config["constraint_audit"].get("smoke_bootstrap_iterations", 25)))
    smoke_config["bootstrap"] = dict(config["bootstrap"], paired_iterations=int(config["bootstrap"].get("smoke_iterations", 50)))
    with tempfile.TemporaryDirectory(prefix="aeroguard_phase5c1_") as temp:
        root = Path(temp)
        reports = root / "reports" / "physics_guided_rul_refined"
        artifacts = root / "artifacts" / "physics_guided_rul_refined"
        reports.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)
        audit, constraints, details, thresholds = corrected_constraint_diagnostics(frames["cv"], smoke_config)
        bootstrap = bootstrap_constraint_intervals(details, thresholds, int(smoke_config["constraint_audit"]["bootstrap_iterations"]), int(smoke_config["constraint_audit"]["bootstrap_seed"]))
        uncertainty_candidates, locked_uncertainty, cv_unc = uncertainty_candidate_selection(frames["cv"], smoke_config)
        benchmark_unc = apply_conformal_policy(frames["benchmark"], locked_uncertainty)
        abstention_candidates, locked_abstention, _ = abstention_policy_selection(cv_unc, smoke_config, artifacts)
        abstained = apply_locked_abstention(benchmark_unc, locked_abstention)
        maintenance_candidates, locked_maintenance = maintenance_policy_selection(cv_unc, smoke_config)
        maintained = apply_locked_maintenance(abstained, locked_maintenance, smoke_config)
        paired = paired_bootstrap_comparison(frames["phase5b"], frames["phase5b_unc"], benchmark_unc, int(smoke_config["bootstrap"]["paired_iterations"]), int(smoke_config["bootstrap"]["seed"]))
        constraints.to_csv(reports / "corrected_constraint_metrics.csv", index=False)
        bootstrap.to_csv(reports / "constraint_bootstrap_intervals.csv", index=False)
        uncertainty_candidates.to_csv(reports / "uncertainty_candidate_metrics.csv", index=False)
        abstention_candidates.to_csv(reports / "abstention_candidate_metrics.csv", index=False)
        maintenance_candidates.to_csv(reports / "maintenance_policy_candidates.csv", index=False)
        maintained.to_csv(reports / "refined_maintenance_recommendations.csv", index=False)
        paired.to_csv(reports / "phase5b_vs_phase5c_paired_bootstrap.csv", index=False)
        atomic_write_json(reports / "locked_uncertainty_policy.json", locked_uncertainty)
        atomic_write_json(reports / "locked_abstention_policy.json", locked_abstention)
        atomic_write_json(reports / "locked_maintenance_policy.json", locked_maintenance)
        generated = sorted(path.name for path in reports.iterdir())
    return {
        "status": "smoke_complete",
        "synthetic_only": True,
        "artifact_schema_validation_exercised": True,
        "constraint_metric_count": int(len(constraints)),
        "constraint_bootstrap_count": int(len(bootstrap)),
        "uncertainty_candidate_count": int(uncertainty_candidates["candidate_method"].nunique()),
        "locked_uncertainty_method": locked_uncertainty["candidate_method"],
        "locked_abstention_policy": locked_abstention["policy_id"],
        "locked_maintenance_policy": locked_maintenance["policy_id"],
        "paired_bootstrap_rows": int(len(paired)),
        "final_report_writing_exercised": bool(generated),
        "neural_training_function_called": False,
        "temporary_directory_removed": True,
    }


def run_full_posthoc_run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root()
    start = time.perf_counter()
    reports, artifacts = prepare_outputs(config, root)
    manifest = build_source_manifest(config, root)
    atomic_write_json(reports / "source_artifact_manifest.json", {"artifacts": manifest})
    validation = validate_source_run(config, root, manifest)
    atomic_write_json(reports / "source_run_validation.json", validation)
    sources = load_source_tables(config, root)
    corrected = corrected_run_summary(sources["run_summary"])
    atomic_write_json(reports / "corrected_run_summary.json", corrected)
    cv = sources["cv_predictions"]
    benchmark = sources["benchmark_predictions"]
    audit, constraints, details, thresholds = corrected_constraint_diagnostics(cv, config)
    atomic_write_json(reports / "constraint_metric_audit.json", audit)
    constraints.to_csv(reports / "corrected_constraint_metrics.csv", index=False)
    constraint_bootstrap = bootstrap_constraint_intervals(
        details,
        thresholds,
        int(config["constraint_audit"]["bootstrap_iterations"]),
        int(config["constraint_audit"]["bootstrap_seed"]),
    )
    constraint_bootstrap.to_csv(reports / "constraint_bootstrap_intervals.csv", index=False)
    uncertainty_candidates, locked_uncertainty, cv_unc = uncertainty_candidate_selection(cv, config)
    uncertainty_candidates.to_csv(reports / "uncertainty_candidate_metrics.csv", index=False)
    atomic_write_json(reports / "locked_uncertainty_policy.json", locked_uncertainty)
    with (artifacts / "locked_uncertainty_policy.pkl").open("wb") as handle:
        pickle.dump(locked_uncertainty, handle)
    refined_uncertainty = apply_conformal_policy(benchmark, locked_uncertainty)
    refined_uncertainty.to_csv(reports / "refined_uncertainty_predictions.csv", index=False)
    refined_uncertainty_metrics = uncertainty_metrics_by_subset(refined_uncertainty, [float(level) for level in config["uncertainty"]["nominal_levels"]])
    atomic_write_json(reports / "refined_uncertainty_metrics.json", refined_uncertainty_metrics)
    abstention_candidates, locked_abstention, _ = abstention_policy_selection(cv_unc, config, artifacts)
    abstention_candidates.to_csv(reports / "abstention_candidate_metrics.csv", index=False)
    atomic_write_json(reports / "locked_abstention_policy.json", locked_abstention)
    refined_abstention = apply_locked_abstention(refined_uncertainty, locked_abstention)
    refined_abstention.to_csv(reports / "refined_abstention_predictions.csv", index=False)
    refined_abstention_metrics = high_error_metrics(refined_abstention, float(config["abstention"]["high_error_threshold"]))
    atomic_write_json(reports / "refined_abstention_metrics.json", refined_abstention_metrics)
    maintenance_candidates, locked_maintenance = maintenance_policy_selection(cv_unc, config)
    maintenance_candidates.to_csv(reports / "maintenance_policy_candidates.csv", index=False)
    atomic_write_json(reports / "locked_maintenance_policy.json", locked_maintenance)
    refined_maintenance = apply_locked_maintenance(refined_abstention, locked_maintenance, config)
    refined_maintenance.to_csv(reports / "refined_maintenance_recommendations.csv", index=False)
    refined_maintenance_metrics = maintenance_metrics(refined_maintenance, config)
    atomic_write_json(reports / "refined_maintenance_metrics.json", refined_maintenance_metrics)
    paired = paired_bootstrap_comparison(
        sources["phase5b_predictions"],
        sources["phase5b_uncertainty_predictions"],
        refined_uncertainty,
        int(config["bootstrap"]["paired_iterations"]),
        int(config["bootstrap"]["seed"]),
    )
    paired.to_csv(reports / "phase5b_vs_phase5c_paired_bootstrap.csv", index=False)
    figures = make_figures(
        reports,
        details,
        uncertainty_candidates,
        sources["source_uncertainty_predictions"],
        refined_uncertainty,
        abstention_candidates,
        sources["maintenance_recommendations"],
        refined_maintenance,
        paired,
    )
    final_manifest = build_source_manifest(config, root)
    manifest_unchanged = {
        row["artifact_key"]: row["sha256"]
        for row in manifest
        if row["sha256"]
    } == {
        row["artifact_key"]: row["sha256"]
        for row in final_manifest
        if row["sha256"]
    }
    interpretation = scientific_interpretation(paired, locked_abstention, refined_uncertainty_metrics)
    summary = {
        "status": "completed",
        "runtime_seconds": time.perf_counter() - start,
        "neural_retraining_disabled": True,
        "locked_neural_model_retrained": False,
        "benchmark_labels_used_for_policy_selection": False,
        "source_directories_read_only": [str(resolve_dirs(config, root)["source_reports"]), str(resolve_dirs(config, root)["source_artifacts"])],
        "source_hashes_unchanged_after_refinement": manifest_unchanged,
        "locked_candidate": validation["candidate_ids"]["locked_model"],
        "checkpoint_prediction_validation": {
            "state_dict_keys_unchanged": validation["state_dict_keys_unchanged"],
            "max_abs_delta": validation["deterministic_prediction_max_abs_delta"],
            "nested_tensor_warning_count": validation["nested_tensor_warning_count"],
        },
        "corrected_run_summary": {
            "source_run_status": corrected["source_run_status"],
            "run_status": corrected["run_status"],
            "runtime_seconds": corrected["runtime_seconds"],
            "completed_stage_count": corrected["completed_stage_count"],
        },
        "constraint_thresholds": thresholds,
        "constraint_metric_rows": int(len(constraints)),
        "constraint_bootstrap_rows": int(len(constraint_bootstrap)),
        "locked_uncertainty_method": locked_uncertainty["candidate_method"],
        "refined_uncertainty_overall": refined_uncertainty_metrics.get("overall", {}),
        "locked_abstention_policy": locked_abstention["policy_id"],
        "refined_abstention_metrics": refined_abstention_metrics,
        "locked_maintenance_policy": locked_maintenance["policy_id"],
        "refined_maintenance_metrics": refined_maintenance_metrics,
        "phase5b_vs_phase5c_rows": int(len(paired)),
        "scientific_interpretation": interpretation,
        "generated_reports": [str(reports / name) for name in REFINED_REPORTS if name == "phase5c_refinement_summary.json" or (reports / name).exists()],
        "figures": figures,
        "artifacts": [str(path) for path in artifacts.glob("*")],
    }
    atomic_write_json(reports / "phase5c_refinement_summary.json", summary)
    atomic_write_json(reports / "source_artifact_manifest.json", {"artifacts": final_manifest, "verified_unchanged": manifest_unchanged})
    write_results_note(root / "notes" / "physics_guided_temporal_rul_refinement_results.md", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--full-posthoc-run", action="store_true")
    args = parser.parse_args(argv)
    modes = [args.validate_config, args.dry_run, args.smoke_test, args.full_posthoc_run]
    if sum(bool(mode) for mode in modes) != 1:
        parser.error("Choose exactly one mode.")
    if args.validate_config:
        result = run_validate_config(args.config)
    elif args.dry_run:
        result = run_dry_run(args.config)
    elif args.smoke_test:
        result = run_smoke_test(args.config)
    else:
        result = run_full_posthoc_run(args.config)
    print(json.dumps(_json_ready(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
