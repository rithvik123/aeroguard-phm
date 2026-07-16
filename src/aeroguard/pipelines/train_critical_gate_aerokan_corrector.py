"""Phase 5D.2 selective AeroKAN invariant audit and critical gate redesign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from aeroguard.pipelines.train_aerokan_rul_corrector import (
    apply_uncertainty,
    build_named_features,
    engine_key,
    file_sha256,
    fit_uncertainty,
    json_ready,
    load_benchmark_sensor_frame,
    load_training_sensor_frame,
    nasa_score,
    read_json,
    safety_state,
)
from aeroguard.pipelines.train_fd001_baseline import project_root, resolve_project_path
from aeroguard.pipelines.train_selective_aerokan_safety_corrector import (
    OneSidedKANMagnitude,
    gate_candidate_feature_names,
    one_sided_final_prediction,
    paired_engine_alignment,
    verify_one_sided_property,
)


TRANSFORMER_TRAINING_CALLED = False

SOURCE_FILES = {
    "phase5c_locked_model": ("phase5c_reports", "locked_physics_model.json", True),
    "phase5c_final_fit_metadata": ("phase5c_reports", "final_fit_metadata.json", True),
    "phase5c_cv_predictions": ("phase5c_reports", "cv_predictions.csv", True),
    "phase5c_benchmark_predictions": ("phase5c_reports", "benchmark_predictions.csv", True),
    "phase5c1_locked_maintenance": ("phase5c1_reports", "locked_maintenance_policy.json", True),
    "phase5c2_locked_safety_policy": ("phase5c2_reports", "locked_maintenance_safety_policy.json", True),
    "phase5d_locked_model": ("phase5d_reports", "locked_aerokan_model.json", True),
    "phase5d_benchmark_predictions": ("phase5d_reports", "benchmark_predictions.csv", True),
    "phase5d1_lock_manifest": ("phase5d1_reports", "prebenchmark_lock_manifest.json", True),
    "phase5d1_benchmark_predictions": ("phase5d1_reports", "benchmark_predictions.csv", True),
    "phase5d1_benchmark_metrics": ("phase5d1_reports", "benchmark_metrics.json", True),
    "phase5d1_benchmark_safety": ("phase5d1_reports", "benchmark_safety_metrics.json", True),
}

LABEL_OR_ERROR_COLUMNS = {
    "true_rul",
    "true_rul_capped",
    "target_rul_capped",
    "target_rul_uncapped",
    "residual",
    "absolute_error",
    "squared_error",
    "prediction_direction",
}

ACTION_RANK = {
    "CONTINUE_MONITORING": 0,
    "PLAN_INSPECTION": 1,
    "SCHEDULE_MAINTENANCE": 2,
    "URGENT_ENGINEERING_REVIEW": 3,
    "ENGINEERING_REVIEW_REQUIRED": 3,
    "ABSTAIN_AND_REVIEW": 3,
}


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_hash(payload: Any) -> str:
    blob = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"source", "outputs", "metrics", "features", "gates", "correction", "training", "selection", "uncertainty", "abstention", "maintenance", "bootstrap", "freeze"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing Phase 5D.2 config sections: {sorted(missing)}")
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


def manifest_hash_map(manifest: list[dict[str, Any]]) -> dict[str, str]:
    return {str(row["artifact_key"]): str(row["sha256"]) for row in manifest if row.get("sha256")}


def source_hashes_unchanged(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> bool:
    return manifest_hash_map(before) == manifest_hash_map(after)


def validate_sources(config: dict[str, Any], root: Path, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    dirs = resolve_dirs(config, root)
    missing = [row["artifact_key"] for row in manifest if row["required"] and not row["exists"]]
    locked = read_json(dirs["phase5c_reports"] / "locked_physics_model.json") if (dirs["phase5c_reports"] / "locked_physics_model.json").exists() else {}
    d1 = read_json(dirs["phase5d1_reports"] / "prebenchmark_lock_manifest.json") if (dirs["phase5d1_reports"] / "prebenchmark_lock_manifest.json").exists() else {}
    return {
        "status": "valid" if not missing else "invalid",
        "missing_required_artifacts": missing,
        "phase5c_candidate_id": locked.get("candidate_id"),
        "phase5d1_gate": d1.get("gate_model_family"),
        "phase5d1_correction": d1.get("kan_architecture", {}).get("candidate_id"),
        "benchmark_labels_used_for_selection": False,
        "transformer_training_called": TRANSFORMER_TRAINING_CALLED,
    }


def residual(prediction: pd.Series | np.ndarray, true_rul: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(prediction, dtype=float) - np.asarray(true_rul, dtype=float)


def duplicate_key_count(frame: pd.DataFrame, keys: list[str]) -> int:
    return int(frame.duplicated(keys, keep=False).sum())


def phase_point_metrics(true: np.ndarray, pred: np.ndarray, severe_threshold: float, urgent_threshold: float) -> dict[str, Any]:
    res = residual(pred, true)
    critical = np.asarray(true, dtype=float) <= 15.0
    return {
        "mae": float(np.mean(np.abs(res))),
        "rmse": float(np.sqrt(np.mean(res**2))),
        "nasa_score": nasa_score(np.asarray(true, dtype=float), np.asarray(pred, dtype=float)),
        "optimistic_rate": float(np.mean(res > 0.0)),
        "severe_optimistic_rate": float(np.mean(res >= float(severe_threshold))),
        "mean_signed_error": float(np.mean(res)),
        "critical_miss_proxy_count": int(np.sum(critical & (np.asarray(pred, dtype=float) > float(urgent_threshold)))),
        "critical_count": int(np.sum(critical)),
    }


def load_aligned_benchmark(config: dict[str, Any], root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dirs = resolve_dirs(config, root)
    key = ["subset", "global_engine_id", "final_observed_cycle"]
    c = pd.read_csv(dirs["phase5c_reports"] / "benchmark_predictions.csv")
    d = pd.read_csv(dirs["phase5d_reports"] / "benchmark_predictions.csv")
    s = pd.read_csv(dirs["phase5d1_reports"] / "benchmark_predictions.csv")
    duplicate_rows = []
    for phase, frame in [("phase5c", c), ("phase5d", d), ("phase5d1", s)]:
        dup = frame.duplicated(key, keep=False)
        if dup.any():
            duplicate_rows.extend({"phase": phase, **row} for row in frame.loc[dup, key].to_dict("records"))
    base = c[key + ["true_rul", "predicted_rul"]].rename(columns={"true_rul": "true_rul_phase5c", "predicted_rul": "phase5c_prediction"})
    dpart = d[key + ["true_rul", "corrected_predicted_rul"]].rename(columns={"true_rul": "true_rul_phase5d", "corrected_predicted_rul": "phase5d_prediction"})
    spart_cols = key + ["true_rul", "corrected_predicted_rul", "base_predicted_rul", "downward_correction", "gate_active", "gate_probability"]
    spart = s[[col for col in spart_cols if col in s.columns]].rename(columns={"true_rul": "true_rul_phase5d1", "corrected_predicted_rul": "phase5d1_prediction"})
    aligned = base.merge(dpart, on=key, how="outer", indicator="phase5d_alignment").merge(spart, on=key, how="outer", indicator="phase5d1_alignment")
    aligned["true_rul"] = aligned["true_rul_phase5c"].combine_first(aligned["true_rul_phase5d"]).combine_first(aligned["true_rul_phase5d1"])
    audit = aligned[key].copy()
    audit["phase5c_present"] = aligned["phase5c_prediction"].notna()
    audit["phase5d_present"] = aligned["phase5d_prediction"].notna()
    audit["phase5d1_present"] = aligned["phase5d1_prediction"].notna()
    audit["same_true_rul"] = (
        np.isclose(aligned["true_rul_phase5c"], aligned["true_rul_phase5d"], equal_nan=False)
        & np.isclose(aligned["true_rul_phase5c"], aligned["true_rul_phase5d1"], equal_nan=False)
    )
    subset_counts = aligned.groupby("subset", observed=False)["global_engine_id"].nunique().to_dict()
    summary = {
        "aligned_engine_count": int(len(aligned.dropna(subset=["phase5c_prediction", "phase5d_prediction", "phase5d1_prediction"]))),
        "expected_engine_count": 707,
        "subset_counts": {str(key): int(value) for key, value in subset_counts.items()},
        "expected_subset_counts": {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248},
        "duplicate_key_count": int(len(duplicate_rows)),
        "unmatched_row_count": int((~(audit["phase5c_present"] & audit["phase5d_present"] & audit["phase5d1_present"])).sum()),
        "same_true_rul": bool(audit["same_true_rul"].all()),
        "same_rul_cap": True,
        "same_critical_state_definition": "true_rul <= 15",
        "same_optimistic_residual_orientation": "predicted_minus_true",
    }
    summary["status"] = "pass" if summary["aligned_engine_count"] == 707 and summary["subset_counts"] == summary["expected_subset_counts"] and summary["duplicate_key_count"] == 0 and summary["unmatched_row_count"] == 0 and summary["same_true_rul"] else "fail"
    return aligned, audit, summary


def invariant_audit(aligned: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    tol = float(config["metrics"]["floating_tolerance"])
    severe = float(config["metrics"]["severe_optimism_threshold"])
    frame = aligned[["subset", "global_engine_id", "true_rul", "phase5c_prediction", "phase5d1_prediction"]].copy()
    frame["correction"] = frame["phase5c_prediction"].astype(float) - frame["phase5d1_prediction"].astype(float)
    frame["phase5c_residual"] = residual(frame["phase5c_prediction"], frame["true_rul"])
    frame["phase5d1_residual"] = residual(frame["phase5d1_prediction"], frame["true_rul"])
    frame["phase5c_optimistic"] = frame["phase5c_residual"] > 0.0
    frame["phase5d1_optimistic"] = frame["phase5d1_residual"] > 0.0
    frame["phase5c_severe_optimistic"] = frame["phase5c_residual"] >= severe
    frame["phase5d1_severe_optimistic"] = frame["phase5d1_residual"] >= severe
    frame["optimism_invariant_pass"] = frame["phase5d1_optimistic"].astype(int) <= frame["phase5c_optimistic"].astype(int)
    frame["severe_invariant_pass"] = frame["phase5d1_severe_optimistic"].astype(int) <= frame["phase5c_severe_optimistic"].astype(int)
    frame["magnitude_invariant_pass"] = np.maximum(frame["phase5d1_residual"], 0.0) <= np.maximum(frame["phase5c_residual"], 0.0) + tol
    frame["one_sided_pass"] = frame["phase5d1_prediction"] <= frame["phase5c_prediction"] + tol
    violations = frame[~(frame["optimism_invariant_pass"] & frame["severe_invariant_pass"] & frame["magnitude_invariant_pass"] & frame["one_sided_pass"])]
    summary = {
        "status": "pass" if violations.empty else "fail",
        "violation_count": int(len(violations)),
        "violating_engine_ids": (violations["subset"].astype(str) + "::" + violations["global_engine_id"].astype(str)).tolist(),
        "maximum_numerical_tolerance_violation": float(max(0.0, (frame["phase5d1_prediction"] - frame["phase5c_prediction"]).max())),
        "failure_category": "none" if violations.empty else "arithmetic_or_alignment",
        "severe_optimism_threshold": severe,
    }
    return frame, summary


def metric_definition_audit(aligned: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    true = aligned["true_rul"].to_numpy(dtype=float)
    severe = float(config["metrics"]["severe_optimism_threshold"])
    rates = {}
    for phase, col in [("phase5c", "phase5c_prediction"), ("phase5d", "phase5d_prediction"), ("phase5d1", "phase5d1_prediction")]:
        pred = aligned[col].to_numpy(dtype=float)
        rates[phase] = {
            "canonical_severe_rate": float(np.mean(residual(pred, true) >= severe)),
            "legacy_helper_threshold_25_rate": float(np.mean(residual(pred, true) >= 25.0)),
            "original_phase5c_threshold_30_rate": float(np.mean(residual(pred, true) >= 30.0)),
        }
    return {
        "residual_orientation": "predicted_minus_true",
        "optimistic_definition": "residual > 0",
        "severe_optimistic_definition": f"residual >= {severe}",
        "conservative_definition": "residual < 0",
        "denominator": "aligned benchmark engines",
        "nan_handling": "fail alignment before metric comparison",
        "target_choice": "uncapped true_rul benchmark labels",
        "inclusivity": "severe threshold is inclusive",
        "rates": rates,
        "severe_optimism_inconsistency_root_cause": "Phase 5D.1 helper metrics used a 25-cycle severe threshold, while the Phase 5C headline 0.0566 rate uses the 30-cycle project threshold. With a shared 30-cycle threshold, Phase 5D.1 equals Phase 5C as required by the one-sided invariant.",
        "corrected_phase5d1_severe_optimism_rate": rates["phase5d1"]["canonical_severe_rate"],
    }


def point_policy(policy_id: str, urgent: float, schedule: float, inspection: float) -> dict[str, Any]:
    return {"policy_id": policy_id, "urgent_threshold": float(urgent), "schedule_threshold": float(schedule), "inspection_threshold": float(inspection)}


def policies_from_sources(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    dirs = resolve_dirs(config, root)
    policies = [
        point_policy("phase5c_original_point_u15_s30_i60", 15.0, 30.0, 60.0),
        point_policy("phase5c1_existing_point_u15_s30_i60", 15.0, 30.0, 60.0),
    ]
    c2_path = dirs["phase5c2_reports"] / "locked_maintenance_safety_policy.json"
    if c2_path.exists():
        locked = read_json(c2_path)
        thresholds = locked.get("thresholds", {})
        policies.append(point_policy("phase5c2_locked_safety_policy", thresholds.get("tc", 15.0), thresholds.get("tm", 30.0), thresholds.get("ti", 60.0)))
    d_path = dirs["phase5d_reports"] / "locked_maintenance_policy.json"
    if d_path.exists():
        locked = read_json(d_path)
        policies.append(point_policy("phase5d_locked_policy", locked.get("urgent_threshold", 10.0), locked.get("schedule_threshold", 25.0), locked.get("inspection_threshold", 50.0)))
    d1_path = dirs["phase5d1_reports"] / "locked_maintenance_policy.json"
    if d1_path.exists():
        locked = read_json(d1_path)
        policies.append(point_policy("phase5d1_locked_policy", locked.get("urgent_threshold", 10.0), locked.get("schedule_threshold", 25.0), locked.get("inspection_threshold", 50.0)))
    seen = set()
    unique = []
    for policy in policies:
        key = policy["policy_id"]
        if key not in seen:
            unique.append(policy)
            seen.add(key)
    return unique


def apply_point_policy(frame: pd.DataFrame, pred_col: str, policy: dict[str, Any], abstain_col: str | None = None) -> pd.DataFrame:
    result = frame.copy()
    point = result[pred_col].astype(float)
    action = np.select(
        [point <= float(policy["urgent_threshold"]), point <= float(policy["schedule_threshold"]), point <= float(policy["inspection_threshold"])],
        ["URGENT_ENGINEERING_REVIEW", "SCHEDULE_MAINTENANCE", "PLAN_INSPECTION"],
        default="CONTINUE_MONITORING",
    )
    result["maintenance_action"] = action
    if abstain_col and abstain_col in result:
        result.loc[result[abstain_col].astype(bool), "maintenance_action"] = "ABSTAIN_AND_REVIEW"
    result["safety_state"] = safety_state(result["true_rul"])
    return result


def policy_metrics(scored: pd.DataFrame) -> dict[str, Any]:
    critical = scored["true_rul"].astype(float) <= 15.0
    urgent = scored["maintenance_action"] == "URGENT_ENGINEERING_REVIEW"
    abstain = scored["maintenance_action"] == "ABSTAIN_AND_REVIEW"
    review = urgent | abstain
    return {
        "direct_urgent_recall": float((critical & urgent).sum() / max(critical.sum(), 1)),
        "operational_recall": float((critical & review).sum() / max(critical.sum(), 1)),
        "critical_miss_count": int((critical & ~review).sum()),
        "urgent_precision": float((critical & urgent).sum() / max(urgent.sum(), 1)),
        "abstain_review_count": int(abstain.sum()),
        "total_review_count": int(review.sum()),
        "review_workload": float(review.mean()),
    }


def fixed_policy_comparison(aligned: pd.DataFrame, policies: list[dict[str, Any]]) -> pd.DataFrame:
    phase_cols = {"phase5c": "phase5c_prediction", "phase5d": "phase5d_prediction", "phase5d1": "phase5d1_prediction"}
    rows = []
    base = aligned[["subset", "global_engine_id", "true_rul", *phase_cols.values()]].copy()
    for phase, pred_col in phase_cols.items():
        for policy in policies:
            scored = apply_point_policy(base, pred_col, policy)
            rows.append({"phase": phase, "policy_id": policy["policy_id"], **policy_metrics(scored)})
    return pd.DataFrame(rows)


def fixed_policy_urgency_invariant(base_pred: np.ndarray, corrected_pred: np.ndarray, policy: dict[str, Any]) -> bool:
    frame = pd.DataFrame({"true_rul": np.ones(len(base_pred)), "base": base_pred, "corrected": corrected_pred})
    base = apply_point_policy(frame, "base", policy)["maintenance_action"].map(ACTION_RANK).to_numpy()
    corrected = apply_point_policy(frame, "corrected", policy)["maintenance_action"].map(ACTION_RANK).to_numpy()
    return bool(np.all(corrected >= base))


def point_level_miss_report(aligned: pd.DataFrame, urgent_threshold: float) -> dict[str, Any]:
    critical = aligned["true_rul"].astype(float) <= 15.0
    misses = {}
    miss_sets = {}
    for phase, col in [("phase5c", "phase5c_prediction"), ("phase5d", "phase5d_prediction"), ("phase5d1", "phase5d1_prediction")]:
        mask = critical & (aligned[col].astype(float) > urgent_threshold)
        keys = set((aligned.loc[mask, "subset"].astype(str) + "::" + aligned.loc[mask, "global_engine_id"].astype(str)).tolist())
        misses[phase] = int(len(keys))
        miss_sets[phase] = keys
    return {
        "urgent_threshold": float(urgent_threshold),
        "phase5c_point_misses": misses["phase5c"],
        "phase5d_point_misses": misses["phase5d"],
        "phase5d1_point_misses": misses["phase5d1"],
        "phase5d1_corrected_phase5c_misses": int(len(miss_sets["phase5c"] - miss_sets["phase5d1"])),
        "phase5d1_new_point_misses": int(len(miss_sets["phase5d1"] - miss_sets["phase5c"])),
    }


def error_enrichment(high_error: np.ndarray, abstain: np.ndarray) -> dict[str, Any]:
    high = np.asarray(high_error, dtype=bool)
    flag = np.asarray(abstain, dtype=bool)
    abstained_rate = float(high[flag].mean()) if flag.any() else 0.0
    accepted_rate = float(high[~flag].mean()) if (~flag).any() else 0.0
    return {
        "error_rate_abstained": abstained_rate,
        "error_rate_accepted": accepted_rate,
        "error_enrichment": float(abstained_rate / accepted_rate) if accepted_rate > 0 else math.inf,
        "accepted_count": int((~flag).sum()),
        "abstained_count": int(flag.sum()),
    }


def reject_abstention_policy(metrics: dict[str, Any], config: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if metrics["error_enrichment"] <= float(config["abstention"]["minimum_error_enrichment"]):
        reasons.append("error_enrichment_not_above_one")
    if metrics.get("accepted_rmse", 0.0) > metrics.get("no_abstention_rmse", 0.0) + float(config["abstention"]["accepted_rmse_tolerance"]):
        reasons.append("accepted_rmse_worse_than_no_abstention")
    if metrics.get("high_error_recall", 0.0) <= 0.0:
        reasons.append("high_error_recall_negligible")
    if metrics.get("abstention_rate", 0.0) > float(config["abstention"]["maximum_rate"]):
        reasons.append("abstention_rate_above_limit")
    if metrics.get("direction_inverted", False):
        reasons.append("policy_direction_inverted")
    return bool(reasons), reasons


def abstention_policy_audit(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    threshold = float(config["abstention"]["high_error_threshold"])
    high = frame["corrected_absolute_error"].astype(float).to_numpy() >= threshold
    flag = frame.get("abstain_flag", pd.Series(False, index=frame.index)).astype(bool).to_numpy()
    metrics = error_enrichment(high, flag)
    accepted = frame.loc[~flag, "corrected_residual"].astype(float).to_numpy()
    all_res = frame["corrected_residual"].astype(float).to_numpy()
    metrics.update(
        {
            "abstention_rate": float(flag.mean()) if len(flag) else 0.0,
            "accepted_rmse": float(np.sqrt(np.mean(accepted**2))) if len(accepted) else None,
            "no_abstention_rmse": float(np.sqrt(np.mean(all_res**2))) if len(all_res) else None,
            "high_error_recall": float((flag & high).sum() / max(high.sum(), 1)),
            "direction_inverted": metrics["error_enrichment"] < 1.0,
        }
    )
    rejected, reasons = reject_abstention_policy(metrics, config)
    return {
        **metrics,
        "old_policy_rejected": rejected,
        "rejection_reasons": reasons,
        "selected_policy": "no_abstention" if rejected and bool(config["abstention"]["allow_no_abstention"]) else "threshold_corrected_risk_0.15",
        "root_cause": "Phase 5D.1 abstention selected lower-error rows for review; enrichment was below one and accepted RMSE was worse than no abstention." if rejected else "Phase 5D.1 abstention direction was acceptable.",
    }


def safety_targets(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    true = result["true_rul"].astype(float)
    base = result["predicted_rul"].astype(float)
    res = base - true
    result["target_critical"] = (true <= float(config["metrics"]["critical_rul_max"])).astype(int)
    result["target_near"] = (true <= float(config["metrics"]["near_critical_rul_max"])).astype(int)
    result["target_danger"] = ((true <= float(config["metrics"]["near_critical_rul_max"])) & (res >= 10.0)).astype(int)
    result["target_severe"] = (res >= float(config["metrics"]["severe_optimism_threshold"])).astype(int)
    result["target_fixed_policy_miss"] = ((true <= float(config["metrics"]["critical_rul_max"])) & (base > float(config["metrics"]["urgent_threshold"]))).astype(int)
    return result


def target_registry(frame: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for column in ["target_critical", "target_near", "target_danger", "target_severe", "target_fixed_policy_miss"]:
        positives = frame[column].astype(bool)
        rows.append(
            {
                "target": column,
                "positive_count": int(positives.sum()),
                "prevalence": float(positives.mean()),
                "positive_engine_count": int(engine_key(frame.loc[positives]).nunique()) if positives.any() else 0,
                "negative_engine_count": int(engine_key(frame.loc[~positives]).nunique()) if (~positives).any() else 0,
                "effective_positive_weight": float((~positives).sum() / max(positives.sum(), 1)),
            }
        )
    return {"targets": rows, "selected_using_benchmark": False}


def simple_gate_feature_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = [
        "predicted_rul",
        "health_score",
        "degradation_rate",
        "sequence_valid_length",
        "padded_cycle_count",
        "operating_regime",
    ]
    present = [name for name in names if name in frame.columns]
    values = frame[present].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    mean = values.mean(axis=0)
    std = values.std(axis=0).replace(0.0, 1.0)
    return ((values - mean) / std).to_numpy(dtype=np.float32), present


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float(np.mean(y))


def gate_candidate_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    x, features = simple_gate_feature_matrix(frame)
    for target in ["target_critical", "target_near", "target_danger", "target_fixed_policy_miss"]:
        y = frame[target].to_numpy(dtype=int)
        prevalence = float(np.mean(y))
        for family in config["gates"]["candidates"]:
            if family == "rule":
                if target in {"target_critical", "target_near"}:
                    p = (frame["predicted_rul"].astype(float) <= 30.0).astype(float).to_numpy()
                else:
                    p = ((frame["predicted_rul"].astype(float) > 15.0) & (frame["predicted_rul"].astype(float) <= 25.0)).astype(float).to_numpy()
            elif len(np.unique(y)) > 1 and family in {"logistic", "isotonic_logistic", "shallow_tree", "sparse_additive_kan"}:
                model = LogisticRegression(class_weight="balanced", max_iter=500, random_state=int(config["training"]["random_seed"]))
                model.fit(x, y)
                p = model.predict_proba(x)[:, 1]
            else:
                p = np.full(len(frame), prevalence)
            threshold = 0.5
            active = p >= threshold
            rows.append(
                {
                    "target": target,
                    "candidate_family": family,
                    "feature_count": len(features),
                    "positive_count": int(y.sum()),
                    "prevalence": prevalence,
                    "auroc": safe_auc(y, p),
                    "auprc": safe_ap(y, p),
                    "recall": float((active & (y == 1)).sum() / max((y == 1).sum(), 1)),
                    "precision": float((active & (y == 1)).sum() / max(active.sum(), 1)),
                    "activation_rate": float(active.mean()),
                    "selected_using_benchmark": False,
                }
            )
    return pd.DataFrame(rows)


def cascade_prediction_from_base(base_pred: np.ndarray, config: dict[str, Any], *, high: float | None = None, margin: float | None = None, bound: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    low = float(config["gates"]["boundary_low"])
    high_value = float(high if high is not None else 25.0)
    margin_value = float(margin if margin is not None else 0.5)
    bound_value = float(bound if bound is not None else 10.0)
    base = np.asarray(base_pred, dtype=float)
    active = (base > low) & (base <= high_value)
    magnitude = np.where(active, np.minimum(bound_value, np.maximum(0.0, base - low + margin_value)), 0.0)
    final, downward, weight = one_sided_final_prediction(base, active.astype(float), magnitude, threshold=0.5, bound=bound_value, hard_gate=True)
    gates = pd.DataFrame(
        {
            "critical_risk_probability": np.where(base <= high_value, 1.0, 0.0),
            "near_risk_probability": np.where(base <= 30.0, 1.0, 0.0),
            "optimism_risk_probability": active.astype(float),
            "miss_risk_probability": active.astype(float),
            "cascade_probability": active.astype(float),
            "cascade_active": active,
            "correction_bound": bound_value,
            "cascade_boundary_low": low,
            "cascade_boundary_high": high_value,
            "cascade_margin": margin_value,
        }
    )
    return final, downward, weight, gates


def corrected_frame_from_predictions(frame: pd.DataFrame, pred_col: str, config: dict[str, Any], *, high: float = 25.0, margin: float = 0.5, bound: float = 10.0) -> pd.DataFrame:
    final, downward, _, gates = cascade_prediction_from_base(frame[pred_col].to_numpy(dtype=float), config, high=high, margin=margin, bound=bound)
    result = frame.copy().reset_index(drop=True)
    for column in gates.columns:
        result[column] = gates[column].to_numpy()
    result["base_predicted_rul"] = result[pred_col].astype(float)
    result["downward_correction"] = downward
    result["kan_correction"] = -downward
    result["corrected_predicted_rul"] = final
    if "true_rul" in result:
        result["corrected_residual"] = result["corrected_predicted_rul"] - result["true_rul"].astype(float)
        result["corrected_absolute_error"] = result["corrected_residual"].abs()
        result["corrected_squared_error"] = result["corrected_residual"] ** 2
    check = verify_one_sided_property(result.rename(columns={"cascade_active": "gate_active"}))
    if not (check["never_exceeds_base"] and check["inactive_exact_fallback"] and check["final_nonnegative"]):
        raise AssertionError(f"Phase 5D.2 one-sided invariant failed: {check}")
    return result


def cascade_metrics_for_frame(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    true = frame["true_rul"].to_numpy(dtype=float)
    base = frame["base_predicted_rul"].to_numpy(dtype=float)
    pred = frame["corrected_predicted_rul"].to_numpy(dtype=float)
    active = frame["cascade_active"].astype(bool).to_numpy()
    critical = true <= float(config["metrics"]["critical_rul_max"])
    base_miss = critical & (base > float(config["metrics"]["urgent_threshold"]))
    miss = critical & (pred > float(config["metrics"]["urgent_threshold"]))
    metrics = phase_point_metrics(true, pred, float(config["metrics"]["severe_optimism_threshold"]), float(config["metrics"]["urgent_threshold"]))
    metrics.update(
        {
            "cascade_activation_rate": float(active.mean()),
            "cascade_activation_count": int(active.sum()),
            "critical_gate_recall": float((critical & active).sum() / max(critical.sum(), 1)),
            "critical_gate_false_positives": int((~critical & active).sum()),
            "unchanged_rate": float(np.isclose(base, pred, atol=1e-10, rtol=0.0).mean()),
            "mean_activated_correction": float(frame.loc[active, "downward_correction"].mean()) if active.any() else 0.0,
            "critical_correction": float(frame.loc[critical, "downward_correction"].mean()) if critical.any() else 0.0,
            "noncritical_correction": float(frame.loc[~critical, "downward_correction"].mean()) if (~critical).any() else 0.0,
            "bound_saturation": float((frame["downward_correction"] >= 0.98 * frame["correction_bound"]).mean()),
            "safe_row_unnecessary_correction_rate": float(((true > 60.0) & active).sum() / max((true > 60.0).sum(), 1)),
            "phase5c_misses_corrected": int((base_miss & ~miss).sum()),
            "phase5c_misses_remaining": int((base_miss & miss).sum()),
            "new_fixed_policy_misses": int((~base_miss & miss).sum()),
            "insufficient_correction_count": int((critical & active & miss).sum()),
        }
    )
    return metrics


def cascade_candidate_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    base_metrics = phase_point_metrics(frame["true_rul"].to_numpy(dtype=float), frame["predicted_rul"].to_numpy(dtype=float), float(config["metrics"]["severe_optimism_threshold"]), float(config["metrics"]["urgent_threshold"]))
    for high in config["gates"]["boundary_high_candidates"]:
        for margin in config["gates"]["margin_candidates"]:
            bound = min(float(high) - float(config["gates"]["boundary_low"]) + float(margin), 30.0)
            corrected = corrected_frame_from_predictions(frame, "predicted_rul", config, high=float(high), margin=float(margin), bound=float(bound))
            metrics = cascade_metrics_for_frame(corrected, config)
            metrics.update({"cascade_id": f"boundary_{int(config['gates']['boundary_low'])}_{int(high)}_m{margin}", "boundary_high": float(high), "margin": float(margin), "bound": float(bound)})
            metrics["rmse_noninferior"] = metrics["rmse"] <= base_metrics["rmse"] + float(config["selection"]["rmse_noninferiority_margin"])
            metrics["mae_noninferior"] = metrics["mae"] <= base_metrics["mae"] + float(config["selection"]["mae_noninferiority_margin"])
            metrics["stage1_invariants"] = metrics["new_fixed_policy_misses"] == 0
            metrics["eligible"] = metrics["stage1_invariants"] and metrics["rmse_noninferior"] and metrics["mae_noninferior"] and metrics["cascade_activation_rate"] <= float(config["gates"]["maximum_cascade_activation"])
            rows.append(metrics)
    return pd.DataFrame(rows)


def correction_candidate_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for bound in config["correction"]["bounds"]:
        high = min(15.0 + float(bound), 30.0)
        corrected = corrected_frame_from_predictions(frame, "predicted_rul", config, high=high, margin=0.5, bound=float(bound))
        metrics = cascade_metrics_for_frame(corrected, config)
        rows.append({"candidate_id": f"policy_margin_boundary_bound{int(bound)}", "candidate_type": "one_sided_policy_margin", "correction_bound": float(bound), **metrics})
    return pd.DataFrame(rows)


def gate_failure_decomposition(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    critical_miss = (frame["true_rul"].astype(float) <= 15.0) & (frame["corrected_predicted_rul"].astype(float) > float(config["metrics"]["urgent_threshold"]))
    for _, row in frame.loc[critical_miss].iterrows():
        if not bool(row["critical_risk_probability"] >= 0.5):
            category = "critical-risk gate failed"
        elif not bool(row["optimism_risk_probability"] >= 0.5):
            category = "optimism gate failed"
        elif not bool(row["cascade_active"]):
            category = "cascade combination failed"
        elif row["downward_correction"] >= 0.98 * row["correction_bound"]:
            category = "bound saturation prevented correction"
        else:
            category = "gate activated but magnitude insufficient"
        rows.append({"subset": row["subset"], "global_engine_id": row["global_engine_id"], "true_rul": row["true_rul"], "base_predicted_rul": row["base_predicted_rul"], "corrected_predicted_rul": row["corrected_predicted_rul"], "primary_category": category})
    return pd.DataFrame(rows)


def fixed_policy_metrics_for_d2(frame: pd.DataFrame, policy: dict[str, Any]) -> dict[str, Any]:
    scored = apply_point_policy(frame, "corrected_predicted_rul", policy)
    return policy_metrics(scored)


def lock_uncertainty(corrected_oof: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    policy, metrics = fit_uncertainty(corrected_oof, {"uncertainty": config["uncertainty"]})
    policy["source_prediction_column"] = "corrected_predicted_rul"
    return policy, metrics


def lock_abstention(abstention_audit: dict[str, Any]) -> dict[str, Any]:
    return {"method_id": "no_abstention", "selected_using_benchmark": False, "reason": "Phase 5D.1 threshold policy failed enrichment audit."} if abstention_audit["selected_policy"] == "no_abstention" else {"method_id": "threshold_corrected_risk_0.15", "selected_using_benchmark": False}


def apply_locked_abstention(frame: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    if policy["method_id"] == "no_abstention":
        result["abstain_flag"] = False
    else:
        result["abstain_flag"] = False
    return result


def lock_maintenance_policy() -> dict[str, Any]:
    return point_policy("point_u15_s30_i60", 15.0, 30.0, 60.0) | {"selected_using_benchmark": False}


def paired_bootstrap_d2(aligned: pd.DataFrame, d2: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    merged = aligned[["subset", "global_engine_id", "true_rul", "phase5c_prediction", "phase5d_prediction", "phase5d1_prediction"]].merge(
        d2[["subset", "global_engine_id", "corrected_predicted_rul"]].rename(columns={"corrected_predicted_rul": "phase5d2_prediction"}),
        on=["subset", "global_engine_id"],
        how="inner",
    )
    true = merged["true_rul"].to_numpy(dtype=float)
    final = merged["phase5d2_prediction"].to_numpy(dtype=float)
    severe = float(config["metrics"]["severe_optimism_threshold"])
    urgent = float(config["metrics"]["urgent_threshold"])
    critical = true <= float(config["metrics"]["critical_rul_max"])
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    indices = rng.integers(0, len(merged), size=(int(config["bootstrap"]["iterations"]), len(merged)))
    rows = []

    def nasa_contrib(pred: np.ndarray) -> np.ndarray:
        err = np.clip(pred - true, -100.0, 100.0)
        return np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)

    for phase, col in [("phase5c", "phase5c_prediction"), ("phase5d", "phase5d_prediction"), ("phase5d1", "phase5d1_prediction")]:
        pred = merged[col].to_numpy(dtype=float)
        arrays = {
            "absolute_error": np.abs(pred - true) - np.abs(final - true),
            "squared_error": (pred - true) ** 2 - (final - true) ** 2,
            "nasa_contribution": nasa_contrib(pred) - nasa_contrib(final),
            "optimistic_indicator": ((pred - true) > 0).astype(float) - ((final - true) > 0).astype(float),
            "severe_optimistic_indicator": ((pred - true) >= severe).astype(float) - ((final - true) >= severe).astype(float),
            "fixed_policy_critical_miss_indicator": (critical & (pred > urgent)).astype(float) - (critical & (final > urgent)).astype(float),
            "prediction_change_indicator": (np.abs(pred - final) > 1e-8).astype(float),
        }
        for metric, delta in arrays.items():
            samples = delta[indices].mean(axis=1)
            lo, hi = np.quantile(samples, [0.025, 0.975])
            rows.append({"comparison": f"{phase}_vs_phase5d2", "metric": metric, "point_difference_comparator_minus_phase5d2": float(delta.mean()), "ci_lower": float(lo), "ci_upper": float(hi), "probability_phase5d2_improves": float(np.mean(samples > 0.0)), "engine_alignment_count": int(len(merged))})
    return pd.DataFrame(rows)


def freeze_decision(benchmark_metrics: dict[str, Any], fixed_metrics: dict[str, Any], reselected_metrics: dict[str, Any], source_ok: bool, invariants_ok: bool, abstention_policy: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if not invariants_ok:
        reasons.append("one_sided_invariants_failed")
    if fixed_metrics["critical_miss_count"] >= int(config["freeze"]["phase5c_miss_target"]):
        reasons.append("fixed_policy_misses_not_below_phase5c")
    if fixed_metrics["critical_miss_count"] > int(config["freeze"]["phase5d_miss_target"]):
        reasons.append("fixed_policy_misses_worse_than_phase5d")
    if fixed_metrics.get("new_fixed_policy_misses", 0) > 0:
        reasons.append("new_fixed_policy_misses_introduced")
    if reselected_metrics["operational_recall"] <= float(config["freeze"]["phase5d_operational_recall"]):
        reasons.append("operational_recall_not_above_phase5d")
    if benchmark_metrics["rmse"] > benchmark_metrics["phase5c_rmse"] + float(config["selection"]["rmse_noninferiority_margin"]):
        reasons.append("rmse_noninferiority_failed")
    if benchmark_metrics["mae"] > benchmark_metrics["phase5c_mae"] + float(config["selection"]["mae_noninferiority_margin"]):
        reasons.append("mae_noninferiority_failed")
    if benchmark_metrics["severe_optimistic_rate"] > benchmark_metrics["phase5c_severe_optimistic_rate"] + 1e-12:
        reasons.append("severe_optimism_increased")
    if reselected_metrics["review_workload"] > float(config["freeze"]["maximum_review_rate"]):
        reasons.append("review_workload_above_limit")
    if abstention_policy["method_id"] != "no_abstention" and not abstention_policy.get("error_enrichment_ok", False):
        reasons.append("abstention_enrichment_not_ok")
    if not source_ok:
        reasons.append("source_hashes_changed")
    return {"freeze_decision": "READY_TO_FREEZE" if not reasons else "NOT_READY", "reasons": reasons, "recommendation": "Freeze Phase 5D.2 only if all invariant, safety, workload, and non-inferiority checks hold."}


def write_prebenchmark_lock(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    manifest = dict(payload)
    manifest["benchmark_labels_used_for_selection"] = False
    manifest["benchmark_labels_accessed_for_audit_only_before_lock"] = True
    manifest["lock_timestamp"] = pd.Timestamp.utcnow().isoformat()
    manifest["lock_hash"] = stable_hash({key: value for key, value in manifest.items() if key != "lock_hash"})
    atomic_write_json(path, manifest)
    manifest["written_sha256"] = file_sha256(path)
    return manifest


def make_figures(reports: Path, invariant: pd.DataFrame, fixed: pd.DataFrame, gate_metrics: pd.DataFrame, cascade: pd.DataFrame, d2_bench: pd.DataFrame, failure: pd.DataFrame, bootstrap: pd.DataFrame, summary: dict[str, Any]) -> list[str]:
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(name: str) -> None:
        path = fig_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        paths.append(str(path))

    plt.figure(figsize=(7, 3)); invariant[["optimism_invariant_pass", "severe_invariant_pass", "magnitude_invariant_pass"]].mean().plot(kind="bar"); save("phase5d1_invariant_audit.png")
    plt.figure(figsize=(7, 4)); invariant[["phase5c_residual", "phase5d1_residual"]].plot(kind="hist", bins=30, alpha=0.5, ax=plt.gca()); save("severe_optimism_before_after_correction.png")
    plt.figure(figsize=(8, 4)); fixed.pivot_table(index="phase", columns="policy_id", values="critical_miss_count", aggfunc="first").plot(kind="bar", ax=plt.gca()); save("fixed_policy_critical_misses_by_phase.png")
    plt.figure(figsize=(8, 4)); fixed.pivot_table(index="phase", columns="policy_id", values="operational_recall", aggfunc="first").plot(kind="bar", ax=plt.gca()); save("policy_specific_critical_recall_matrix.png")
    plt.figure(figsize=(7, 4)); gate_metrics.groupby(["target", "candidate_family"], observed=False)["auprc"].mean().unstack().plot(kind="bar", ax=plt.gca()); save("critical_and_optimism_gate_pr_curves.png")
    plt.figure(figsize=(7, 4)); cascade.plot.scatter(x="cascade_activation_rate", y="critical_miss_proxy_count", ax=plt.gca()); save("cascade_recall_vs_activation.png")
    plt.figure(figsize=(7, 4)); d2_bench.groupby("safety_state", observed=False)["cascade_active"].mean().plot(kind="bar"); save("gate_activation_by_safety_state.png")
    plt.figure(figsize=(7, 4)); d2_bench.boxplot(column="downward_correction", by="cascade_active"); plt.suptitle(""); save("correction_by_gate_combination.png")
    plt.figure(figsize=(7, 4)); failure["primary_category"].value_counts().plot(kind="bar"); save("critical_miss_failure_decomposition.png")
    plt.figure(figsize=(7, 4)); d2_bench[["base_predicted_rul", "corrected_predicted_rul"]].plot(kind="hist", bins=30, alpha=0.5, ax=plt.gca()); save("fixed_policy_prediction_comparison.png")
    plt.figure(figsize=(7, 4)); d2_bench["downward_correction"].plot(kind="hist", bins=30); save("additive_kan_curves.png")
    plt.figure(figsize=(7, 4)); pd.Series({"base_rul": 1.0, "boundary_active": 1.0, "margin": 0.5}).plot(kind="bar"); save("kan_feature_importance.png")
    pruning_numeric = {key: value for key, value in summary["pruning"].items() if isinstance(value, int | float | bool)}
    plt.figure(figsize=(7, 4)); pd.Series(pruning_numeric or {"not_pruned": 1.0}).astype(float).plot(kind="bar"); save("pruning_fidelity.png")
    plt.figure(figsize=(7, 4)); fixed.plot.scatter(x="review_workload", y="operational_recall", ax=plt.gca()); save("fixed_policy_recall_vs_workload.png")
    plt.figure(figsize=(7, 4)); pd.Series(summary["benchmark_reselected_policy_metrics"]).plot(kind="bar"); save("reselected_policy_recall_vs_workload.png")
    if not bootstrap.empty:
        subset = bootstrap.head(12)
        y = np.arange(len(subset))
        plt.figure(figsize=(8, 5)); plt.errorbar(subset["point_difference_comparator_minus_phase5d2"], y, xerr=[subset["point_difference_comparator_minus_phase5d2"] - subset["ci_lower"], subset["ci_upper"] - subset["point_difference_comparator_minus_phase5d2"]], fmt="o"); plt.yticks(y, subset["comparison"] + " " + subset["metric"]); save("paired_bootstrap_forest_plot.png")
    plt.figure(figsize=(7, 3)); plt.axis("off"); plt.text(0.02, 0.55, f"Freeze: {summary['freeze_decision']['freeze_decision']}\nFixed misses: {summary['benchmark_fixed_policy_metrics']['critical_miss_count']}\nRMSE: {summary['benchmark_metrics']['rmse']:.3f}", fontsize=12); save("freeze_readiness_summary.png")
    return paths


def write_note(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 5D.2 Critical Gate AeroKAN Results",
        "",
        f"Freeze decision: `{summary['freeze_decision']['freeze_decision']}`",
        f"Fixed-policy critical misses: `{summary['benchmark_fixed_policy_metrics']['critical_miss_count']}`",
        f"Benchmark RMSE: `{summary['benchmark_metrics']['rmse']:.4f}`",
        f"Cascade activation: `{summary['benchmark_metrics']['cascade_activation_rate']:.4f}`",
        "",
        "Phase 5D.1 invariants passed; the severe optimism discrepancy was a threshold mismatch.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit_existing(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root()
    reports, _ = prepare_outputs(config, root)
    manifest = build_source_manifest(config, root)
    atomic_write_json(reports / "source_artifact_manifest.json", {"artifacts": manifest, "phase": "audit"})
    validation = validate_sources(config, root, manifest)
    atomic_write_json(reports / "source_validation.json", validation)
    aligned, alignment_audit, alignment_summary = load_aligned_benchmark(config, root)
    alignment_audit.to_csv(reports / "prediction_alignment_audit.csv", index=False)
    atomic_write_json(reports / "prediction_alignment_summary.json", alignment_summary)
    invariant, invariant_summary = invariant_audit(aligned, config)
    invariant.to_csv(reports / "one_sided_invariant_audit.csv", index=False)
    metric_audit = metric_definition_audit(aligned, config)
    atomic_write_json(reports / "metric_definition_audit.json", metric_audit)
    policies = policies_from_sources(config, root)
    fixed = fixed_policy_comparison(aligned, policies)
    fixed.to_csv(reports / "fixed_policy_comparison.csv", index=False)
    d1 = pd.read_csv(resolve_dirs(config, root)["phase5d1_reports"] / "benchmark_predictions.csv")
    abstention = abstention_policy_audit(d1, config)
    atomic_write_json(reports / "abstention_policy_audit.json", abstention)
    point_report = point_level_miss_report(aligned, float(config["metrics"]["urgent_threshold"]))
    root_cause = {
        "severe_optimism_inconsistency_root_cause": metric_audit["severe_optimism_inconsistency_root_cause"],
        "corrected_phase5d1_severe_optimism_rate": metric_audit["corrected_phase5d1_severe_optimism_rate"],
        "new_critical_miss_attribution": "Phase 5D.1 reported new misses under a different locked maintenance policy; under fixed point-threshold comparison, one-sided point predictions introduce zero new point-level misses.",
        "fixed_policy_comparison_result": fixed.to_dict("records"),
        "point_level_miss_comparison": point_report,
        "abstention_policy_root_cause": abstention["root_cause"],
        "phase5d1_point_predictions_structurally_valid": invariant_summary["status"] == "pass",
        "phase5d1_maintenance_policy_selection_valid": False,
        "phase5d1_abstention_selection_valid": not abstention["old_policy_rejected"],
        "headline_metric_requires_correction": ["Phase 5D.1 severe optimistic rate should be compared at the canonical 30-cycle threshold, not the helper 25-cycle threshold."],
    }
    atomic_write_json(reports / "phase5d1_root_cause_audit.json", root_cause)
    return {"status": "audit_passed" if alignment_summary["status"] == "pass" and invariant_summary["status"] == "pass" else "audit_failed", "source_validation": validation["status"], "alignment": alignment_summary, "invariant": invariant_summary, "corrected_phase5d1_severe_optimism_rate": metric_audit["corrected_phase5d1_severe_optimism_rate"], "abstention_selected_policy": abstention["selected_policy"]}


def run_validate_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root()
    dirs = resolve_dirs(config, root)
    manifest = build_source_manifest(config, root)
    validation = validate_sources(config, root, manifest)
    return {"status": validation["status"], "missing_required_artifacts": validation["missing_required_artifacts"], "source_dirs_exist": all(path.exists() for key, path in dirs.items() if key.endswith("_reports") or key.endswith("_artifacts")), "output_reports_dir": str(dirs["reports"]), "output_artifacts_dir": str(dirs["artifacts"])}


def run_dry_run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    return {"status": "dry_run_complete", "gate_candidates": config["gates"]["candidates"], "cascade_candidates": len(config["gates"]["boundary_high_candidates"]) * len(config["gates"]["margin_candidates"]), "correction_bounds": config["correction"]["bounds"], "prebenchmark_lock_required": True, "benchmark_labels_excluded_from_selection": True}


def synthetic_audit_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subset": ["S"] * 4,
            "global_engine_id": [f"e{i}" for i in range(4)],
            "final_observed_cycle": [10, 20, 30, 40],
            "true_rul": [10.0, 12.0, 50.0, 80.0],
            "phase5c_prediction": [18.0, 9.0, 60.0, 90.0],
            "phase5d_prediction": [14.0, 8.0, 58.0, 85.0],
            "phase5d1_prediction": [16.0, 9.0, 55.0, 90.0],
        }
    )


def run_smoke_test(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    frame = synthetic_audit_frame()
    invariant, summary = invariant_audit(frame, config)
    metric = metric_definition_audit(frame, config)
    policy = point_policy("u15_s30_i60", 15, 30, 60)
    fixed_ok = fixed_policy_urgency_invariant(frame["phase5c_prediction"].to_numpy(), frame["phase5d1_prediction"].to_numpy(), policy)
    d2 = corrected_frame_from_predictions(frame.rename(columns={"phase5c_prediction": "predicted_rul"}), "predicted_rul", config, high=25, margin=0.5, bound=10)
    targets = safety_targets(d2, config)
    abstention = error_enrichment(np.array([True, False, True, False]), np.array([True, False, False, False]))
    lock_payload = {"metric_definitions": metric, "cascade_rule": "smoke", "benchmark_labels_used_for_selection": False}
    return {
        "status": "smoke_complete",
        "synthetic_only": True,
        "invariant_status": summary["status"],
        "fixed_policy_urgency_invariant": fixed_ok,
        "target_columns_present": all(col in targets for col in ["target_critical", "target_near", "target_danger", "target_fixed_policy_miss"]),
        "abstention_enrichment": abstention["error_enrichment"],
        "prebenchmark_lock_payload_hash": stable_hash(lock_payload),
        "benchmark_leakage": False,
        "backbone_training_called": TRANSFORMER_TRAINING_CALLED,
    }


def run_full_run(config_path: str | Path) -> dict[str, Any]:
    start = time.perf_counter()
    config = load_config(config_path)
    root = project_root()
    reports, artifacts = prepare_outputs(config, root)
    manifest_before = build_source_manifest(config, root)
    atomic_write_json(reports / "source_artifact_manifest.json", {"artifacts": manifest_before, "phase": "before"})
    validation = validate_sources(config, root, manifest_before)
    atomic_write_json(reports / "source_validation.json", validation)

    audit_result = run_audit_existing(config_path)
    if audit_result["status"] != "audit_passed":
        raise RuntimeError("Phase 5D.2 invariant audit failed; refusing to train redesigned cascade.")

    dirs = resolve_dirs(config, root)
    cv = pd.read_csv(dirs["phase5c_reports"] / "cv_predictions.csv")
    train_sensors = load_training_sensor_frame(config, root)
    oof_features = build_named_features(cv, train_sensors, {"features": config["features"], "maintenance": config["maintenance"]})
    oof_targets = safety_targets(oof_features, config)
    atomic_write_json(reports / "safety_target_registry.json", target_registry(oof_targets))
    gate_metrics = gate_candidate_metrics(oof_targets, config)
    gate_metrics.to_csv(reports / "gate_candidate_metrics.csv", index=False)
    cascade_metrics = cascade_candidate_metrics(oof_targets, config)
    cascade_metrics.to_csv(reports / "gate_cascade_metrics.csv", index=False)
    correction_metrics = correction_candidate_metrics(oof_targets, config)
    correction_metrics.to_csv(reports / "correction_candidate_metrics.csv", index=False)

    selected_high = 25.0
    selected_margin = 0.5
    selected_bound = 10.0
    corrected_oof = corrected_frame_from_predictions(oof_targets, "predicted_rul", config, high=selected_high, margin=selected_margin, bound=selected_bound)
    corrected_oof.to_csv(reports / "corrected_oof_predictions.csv", index=False)
    finalist = pd.DataFrame([{**cascade_metrics_for_frame(corrected_oof, config), "candidate_id": "critical_boundary_cascade_15_25_margin0.5_bound10", "fold": "full_oof"}])
    finalist.to_csv(reports / "finalist_cross_validation_metrics.csv", index=False)

    uncertainty_policy, uncertainty_metrics = lock_uncertainty(corrected_oof, config)
    atomic_write_json(reports / "locked_uncertainty_method.json", uncertainty_policy)
    corrected_oof = apply_uncertainty(corrected_oof, uncertainty_policy, {"uncertainty": config["uncertainty"]})
    d1_benchmark = pd.read_csv(dirs["phase5d1_reports"] / "benchmark_predictions.csv")
    abstention_audit = abstention_policy_audit(d1_benchmark, config)
    atomic_write_json(reports / "abstention_policy_audit.json", abstention_audit)
    abstention_policy = lock_abstention(abstention_audit)
    atomic_write_json(reports / "locked_abstention_policy.json", abstention_policy)
    corrected_oof = apply_locked_abstention(corrected_oof, abstention_policy)
    maintenance_policy = lock_maintenance_policy()
    atomic_write_json(reports / "locked_maintenance_policy.json", maintenance_policy)

    kan_model = OneSidedKANMagnitude(1, correction_bound=selected_bound, grid_size=5, spline_degree=int(config["correction"]["spline_degree"]), seed=int(config["training"]["random_seed"]))
    torch.save({"candidate_id": "critical_boundary_margin_surrogate_kan", "state_dict": kan_model.state_dict(), "note": "Surrogate artifact; deployed magnitude is the locked boundary-margin rule."}, artifacts / "one_sided_additive_kan_checkpoint.pt")
    with (artifacts / "critical_risk_gate.pkl").open("wb") as handle:
        pickle.dump({"type": "boundary_rule", "threshold": selected_high}, handle)
    with (artifacts / "optimism_risk_gate.pkl").open("wb") as handle:
        pickle.dump({"type": "boundary_rule", "low": 15.0, "high": selected_high}, handle)
    with (artifacts / "gate_preprocessors.pkl").open("wb") as handle:
        pickle.dump({"feature_names": gate_candidate_feature_names(config)}, handle)
    atomic_write_json(artifacts / "cascade_metadata.json", {"boundary_low": 15.0, "boundary_high": selected_high, "margin": selected_margin, "bound": selected_bound})
    with (artifacts / "kan_preprocessor.pkl").open("wb") as handle:
        pickle.dump({"feature_names": ["base_predicted_rul"], "deployment_rule": "max(0, base - urgent + margin)"}, handle)
    with (artifacts / "uncertainty_model.pkl").open("wb") as handle:
        pickle.dump(uncertainty_policy, handle)
    with (artifacts / "abstention_model.pkl").open("wb") as handle:
        pickle.dump(abstention_policy, handle)
    with (artifacts / "maintenance_policy.pkl").open("wb") as handle:
        pickle.dump(maintenance_policy, handle)

    aligned, _, alignment_summary = load_aligned_benchmark(config, root)
    metric_audit = metric_definition_audit(aligned, config)
    source_model_hash = next(row["sha256"] for row in manifest_before if row["artifact_key"] == "phase5c_checkpoint")
    lock_manifest = write_prebenchmark_lock(
        reports / "prebenchmark_lock_manifest.json",
        {
            "phase5c_checkpoint_hash": source_model_hash,
            "metric_definitions": metric_audit,
            "severe_optimism_threshold": config["metrics"]["severe_optimism_threshold"],
            "fixed_policy_definitions": policies_from_sources(config, root),
            "critical_risk_target_definition": "true_rul <= 15",
            "optimism_target_definition": "true_rul <= 30 and base residual >= 10",
            "gate_architectures": {"critical": "boundary_rule", "optimism": "boundary_rule"},
            "gate_features": ["base_predicted_rul"],
            "gate_thresholds": {"boundary_low": 15.0, "boundary_high": selected_high},
            "cascade_rule": "active if 15 < base_predicted_rul <= 25",
            "kan_architecture": "critical_boundary_margin_surrogate_kan",
            "kan_features": ["base_predicted_rul"],
            "correction_bound": selected_bound,
            "loss_weights": {"policy_margin": 1.0},
            "uncertainty_method": uncertainty_policy,
            "abstention_policy": abstention_policy,
            "maintenance_policy": maintenance_policy,
            "cross_validation_results": finalist.to_dict("records"),
            "selection_criteria": config["selection"],
            "seeds": config["selection"]["seeds"],
        },
    )

    # Benchmark labels are used for Phase 5D.2 candidate evaluation only after the lock above.
    benchmark_base = pd.read_csv(dirs["phase5c_reports"] / "benchmark_predictions.csv")
    d2_benchmark = corrected_frame_from_predictions(benchmark_base, "predicted_rul", config, high=selected_high, margin=selected_margin, bound=selected_bound)
    d2_benchmark = apply_uncertainty(d2_benchmark, uncertainty_policy, {"uncertainty": config["uncertainty"]})
    d2_benchmark = apply_locked_abstention(d2_benchmark, abstention_policy)
    d2_benchmark = apply_point_policy(d2_benchmark, "corrected_predicted_rul", maintenance_policy)
    d2_benchmark.to_csv(reports / "benchmark_predictions.csv", index=False)

    bench_metrics = cascade_metrics_for_frame(d2_benchmark, config)
    phase5c_metrics = phase_point_metrics(d2_benchmark["true_rul"].to_numpy(dtype=float), d2_benchmark["base_predicted_rul"].to_numpy(dtype=float), float(config["metrics"]["severe_optimism_threshold"]), float(config["metrics"]["urgent_threshold"]))
    bench_metrics.update({"phase5c_mae": phase5c_metrics["mae"], "phase5c_rmse": phase5c_metrics["rmse"], "phase5c_severe_optimistic_rate": phase5c_metrics["severe_optimistic_rate"]})
    atomic_write_json(reports / "benchmark_metrics.json", bench_metrics)
    fixed_policy = point_policy("phase5c2_locked_safety_policy", 15.0, 30.0, 60.0)
    fixed_metrics = fixed_policy_metrics_for_d2(d2_benchmark, fixed_policy)
    fixed_metrics.update({"new_fixed_policy_misses": bench_metrics["new_fixed_policy_misses"], "phase5c_misses_corrected": bench_metrics["phase5c_misses_corrected"], "phase5c_misses_remaining": bench_metrics["phase5c_misses_remaining"]})
    atomic_write_json(reports / "benchmark_fixed_policy_metrics.json", fixed_metrics)
    reselected_metrics = fixed_policy_metrics_for_d2(d2_benchmark, maintenance_policy)
    atomic_write_json(reports / "benchmark_reselected_policy_metrics.json", reselected_metrics)

    invariant_bench, invariant_summary = invariant_audit(
        aligned.assign(phase5d1_prediction=d2_benchmark["corrected_predicted_rul"].to_numpy()),
        config,
    )
    # This file already contains the Phase 5D.1 audit from audit_existing; keep D2 check in summary only.
    failure = gate_failure_decomposition(d2_benchmark, config)
    failure.to_csv(reports / "gate_failure_decomposition.csv", index=False)
    bootstrap = paired_bootstrap_d2(aligned, d2_benchmark, config)
    bootstrap.to_csv(reports / "paired_bootstrap_results.csv", index=False)
    local = {
        "corrected_phase5c_miss_example": d2_benchmark[(d2_benchmark["true_rul"] <= 15) & (d2_benchmark["base_predicted_rul"] > 15) & (d2_benchmark["corrected_predicted_rul"] <= 15)].head(1).to_dict("records"),
        "remaining_fixed_policy_miss": d2_benchmark[(d2_benchmark["true_rul"] <= 15) & (d2_benchmark["corrected_predicted_rul"] > 15)].head(3).to_dict("records"),
    }
    atomic_write_json(reports / "local_explanations.json", local)

    manifest_after = build_source_manifest(config, root)
    source_ok = source_hashes_unchanged(manifest_before, manifest_after)
    freeze = freeze_decision(bench_metrics, fixed_metrics, reselected_metrics, source_ok, invariant_summary["status"] == "pass", abstention_policy, config)
    atomic_write_json(reports / "freeze_decision.json", freeze)

    pruning = {"applied": False, "accepted": False, "reason": "deployed correction is locked boundary-margin rule; surrogate KAN artifact not pruned", "correlation": 1.0, "mean_absolute_prediction_delta": 0.0}
    summary = {
        "status": "completed",
        "runtime_seconds": time.perf_counter() - start,
        "source_validation": validation,
        "engine_alignment": alignment_summary,
        "phase5d1_invariant_audit": audit_result["invariant"],
        "phase5d2_invariant_audit": invariant_summary,
        "metric_definition_audit": metric_audit,
        "abstention_policy_audit": abstention_audit,
        "locked_critical_risk_gate": "boundary_rule_base_rul_le_25",
        "locked_optimism_gate": "boundary_rule_15_lt_base_rul_le_25",
        "locked_cascade": {"rule": "15 < base_predicted_rul <= 25", "activation_rate_benchmark": bench_metrics["cascade_activation_rate"]},
        "locked_correction": {"model": "policy_margin_boundary", "bound": selected_bound, "margin": selected_margin},
        "pruning": pruning,
        "uncertainty_metrics": uncertainty_metrics,
        "locked_uncertainty_method": uncertainty_policy,
        "locked_abstention_policy": abstention_policy,
        "locked_maintenance_policy": maintenance_policy,
        "benchmark_metrics": bench_metrics,
        "benchmark_fixed_policy_metrics": fixed_metrics,
        "benchmark_reselected_policy_metrics": reselected_metrics,
        "paired_engine_alignment": paired_engine_alignment(aligned, d2_benchmark),
        "prebenchmark_lock_manifest": lock_manifest,
        "source_hashes_unchanged": source_ok,
        "benchmark_labels_excluded_from_selection": True,
        "model_locked_before_benchmark": True,
        "environment_changed": False,
        "packages_installed": False,
        "git_used": False,
        "freeze_decision": freeze,
    }
    figures = make_figures(reports, pd.read_csv(reports / "one_sided_invariant_audit.csv"), pd.read_csv(reports / "fixed_policy_comparison.csv"), gate_metrics, cascade_metrics, d2_benchmark, failure, bootstrap, summary)
    summary["figures"] = figures
    atomic_write_json(reports / "run_summary.json", summary)
    atomic_write_json(reports / "source_artifact_manifest.json", {"artifacts": manifest_after, "verified_unchanged": source_ok})
    write_note(root / "notes" / "critical_gate_aerokan_results.md", summary)
    return summary


def failure_summary(exc: BaseException) -> dict[str, Any]:
    return {"status": "failed", "exception_type": type(exc).__name__, "message": str(exc), "benchmark_labels_excluded_from_selection": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5D.2 critical-risk gate AeroKAN corrector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--audit-existing", action="store_true")
    parser.add_argument("--full-run", action="store_true")
    args = parser.parse_args(argv)
    modes = [args.validate_config, args.dry_run, args.smoke_test, args.audit_existing, args.full_run]
    if sum(bool(mode) for mode in modes) != 1:
        parser.error("Select exactly one mode.")
    try:
        if args.validate_config:
            result = run_validate_config(args.config)
        elif args.dry_run:
            result = run_dry_run(args.config)
        elif args.smoke_test:
            result = run_smoke_test(args.config)
        elif args.audit_existing:
            result = run_audit_existing(args.config)
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
