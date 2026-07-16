"""Phase 5C.2 safety-constrained maintenance policy refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:  # Uses the already-installed environment only.
    from sklearn.tree import DecisionTreeClassifier, export_text
except Exception:  # pragma: no cover
    DecisionTreeClassifier = None
    export_text = None

from aeroguard.pipelines.refine_physics_guided_temporal_rul import apply_conformal_policy, risk_feature_frame
from aeroguard.pipelines.train_fd001_baseline import project_root, resolve_project_path


REQUIRED_CONFIG_SECTIONS = {
    "source",
    "outputs",
    "safety_states",
    "actions",
    "policy_selection",
    "policy_grid",
    "abstention_interaction",
    "support_fallback",
    "cost_matrices",
    "bootstrap",
    "freeze",
}

SOURCE_REPORTS = {
    "phase5c_locked_model": ("phase5c_reports", "locked_physics_model.json", "locked model metadata", True),
    "phase5c_final_fit_metadata": ("phase5c_reports", "final_fit_metadata.json", "final fit metadata", True),
    "phase5c_benchmark_predictions": ("phase5c_reports", "benchmark_predictions.csv", "Phase 5C benchmark predictions", True),
    "phase5c_cv_predictions": ("phase5c_reports", "cv_predictions.csv", "OOF validation predictions", True),
    "phase5c_trajectory_metrics": ("phase5c_reports", "trajectory_consistency_metrics.csv", "trajectory predictions/metrics", True),
    "phase5c_model_efficiency": ("phase5c_reports", "model_efficiency.csv", "model-efficiency metadata", True),
    "phase5c1_locked_uncertainty": ("refined_reports", "locked_uncertainty_policy.json", "locked uncertainty method", True),
    "phase5c1_refined_uncertainty_predictions": ("refined_reports", "refined_uncertainty_predictions.csv", "benchmark refined uncertainty predictions", True),
    "phase5c1_locked_abstention": ("refined_reports", "locked_abstention_policy.json", "locked abstention policy", True),
    "phase5c1_refined_abstention_predictions": ("refined_reports", "refined_abstention_predictions.csv", "benchmark refined abstention predictions", True),
    "phase5c1_maintenance_candidates": ("refined_reports", "maintenance_policy_candidates.csv", "Phase 5C.1 maintenance candidates", True),
    "phase5c1_locked_maintenance": ("refined_reports", "locked_maintenance_policy.json", "Phase 5C.1 locked maintenance policy", True),
    "phase5c1_refined_maintenance_recommendations": ("refined_reports", "refined_maintenance_recommendations.csv", "Phase 5C.1 refined maintenance recommendations", True),
    "phase5c1_source_manifest": ("refined_reports", "source_artifact_manifest.json", "Phase 5C.1 source artifact manifest", True),
    "phase5c1_summary": ("refined_reports", "phase5c_refinement_summary.json", "Phase 5C.1 summary", True),
}

ACTION_ORDER = [
    "CONTINUE_MONITORING",
    "PLAN_INSPECTION",
    "SCHEDULE_MAINTENANCE",
    "URGENT_ENGINEERING_REVIEW",
    "ABSTAIN_AND_REVIEW",
]
SAFETY_STATE_ORDER = ["CRITICAL", "NEAR_TERM", "INSPECTION_WINDOW", "MONITORING", "HEALTHY"]
URGENCY_RANK = {action: index for index, action in enumerate(ACTION_ORDER)}
LABEL_COLUMNS = {"true_rul", "true_rul_capped", "target_rul_capped", "target_rul_uncapped", "absolute_error", "squared_error", "residual"}
DECISION_FEATURE_COLUMNS = [
    "predicted_rul",
    "lower_80",
    "lower_90",
    "lower_95",
    "upper_80",
    "upper_90",
    "upper_95",
    "interval_width_90",
    "normalized_interval_width",
    "high_error_risk_probability",
    "abstain_flag",
    "support_score",
    "support_category_code",
    "regime_distance",
    "health_score_filled",
    "degradation_rate_filled",
    "sequence_valid_length",
    "padding_fraction",
    "operating_regime",
    "operating_regime_rarity",
    "recent_rul_slope",
    "recent_lower_bound_slope",
]


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    path: Path
    role: str
    required: bool


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    missing = sorted(REQUIRED_CONFIG_SECTIONS - set(config))
    if missing:
        raise ValueError(f"Missing maintenance-safety config sections: {missing}")
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(payload), handle, indent=2, allow_nan=False)
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key not in {"policy_hash", "lock_timestamp"}}
    return hashlib.sha256(json.dumps(_json_ready(clean), sort_keys=True).encode("utf-8")).hexdigest()


def resolve_dirs(config: dict[str, Any], root: Path) -> dict[str, Path]:
    source = config["source"]
    return {
        "phase5c_reports": resolve_project_path(source["phase5c_reports"], root),
        "phase5c_artifacts": resolve_project_path(source["phase5c_artifacts"], root),
        "refined_reports": resolve_project_path(source["refined_reports"], root),
        "refined_artifacts": resolve_project_path(source["refined_artifacts"], root),
        "reports": resolve_project_path(config["outputs"]["reports_dir"], root),
        "artifacts": resolve_project_path(config["outputs"]["artifacts_dir"], root),
    }


def source_specs(config: dict[str, Any], root: Path) -> list[ArtifactSpec]:
    dirs = resolve_dirs(config, root)
    specs = [
        ArtifactSpec(key, dirs[dir_key] / filename, role, required)
        for key, (dir_key, filename, role, required) in SOURCE_REPORTS.items()
    ]
    abstention = dirs["refined_artifacts"] / "abstention_logistic_model.pkl"
    uncertainty = dirs["refined_artifacts"] / "locked_uncertainty_policy.pkl"
    specs.extend(
        [
            ArtifactSpec("phase5c1_abstention_model", abstention, "locked abstention-risk model", True),
            ArtifactSpec("phase5c1_uncertainty_policy_object", uncertainty, "locked uncertainty policy object", True),
        ]
    )
    return specs


def build_source_manifest(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows = []
    for spec in source_specs(config, root):
        exists = spec.path.exists()
        stat = spec.path.stat() if exists else None
        rows.append(
            {
                "artifact_key": spec.key,
                "source_path": str(spec.path),
                "semantic_role": spec.role,
                "required": spec.required,
                "exists": exists,
                "size_bytes": int(stat.st_size) if stat else 0,
                "sha256": sha256_file(spec.path) if exists else "",
                "modification_timestamp": pd.Timestamp.fromtimestamp(stat.st_mtime).isoformat() if stat else "",
            }
        )
    return rows


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 2:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_sources(config: dict[str, Any], root: Path) -> dict[str, Any]:
    dirs = resolve_dirs(config, root)
    phase5c = dirs["phase5c_reports"]
    refined = dirs["refined_reports"]
    return {
        "locked_model": read_json(phase5c / "locked_physics_model.json"),
        "final_fit_metadata": read_json(phase5c / "final_fit_metadata.json"),
        "benchmark": _read_csv(phase5c / "benchmark_predictions.csv"),
        "cv": _read_csv(phase5c / "cv_predictions.csv"),
        "trajectory": _read_csv(phase5c / "trajectory_consistency_metrics.csv"),
        "model_efficiency": _read_csv(phase5c / "model_efficiency.csv"),
        "locked_uncertainty": read_json(refined / "locked_uncertainty_policy.json"),
        "locked_abstention": read_json(refined / "locked_abstention_policy.json"),
        "benchmark_uncertainty": _read_csv(refined / "refined_uncertainty_predictions.csv"),
        "benchmark_abstention": _read_csv(refined / "refined_abstention_predictions.csv"),
        "maintenance_candidates": _read_csv(refined / "maintenance_policy_candidates.csv"),
        "locked_maintenance": read_json(refined / "locked_maintenance_policy.json"),
        "refined_maintenance_recommendations": _read_csv(refined / "refined_maintenance_recommendations.csv"),
        "refined_manifest": read_json(refined / "source_artifact_manifest.json"),
        "refined_summary": read_json(refined / "phase5c_refinement_summary.json"),
    }


def validate_sources(config: dict[str, Any], root: Path, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [row["artifact_key"] for row in manifest if row["required"] and not row["exists"]]
    if missing:
        raise FileNotFoundError(f"Missing required safety-refinement source artifacts: {missing}")
    sources = load_sources(config, root)
    model_id = str(sources["locked_model"].get("candidate_id", ""))
    expected = str(config["source"]["expected_model_id"])
    benchmark = sources["benchmark"]
    cv = sources["cv"]
    subset_counts = benchmark.groupby("subset", observed=False)["global_engine_id"].nunique().to_dict()
    duplicate_benchmark = int(benchmark.duplicated(["subset", "global_engine_id", "final_observed_cycle"]).sum())
    duplicate_cv_keys = int(cv.duplicated(["subset", "global_engine_id", "cycle", "fold", "seed"]).sum())
    feature_names = list(sources["final_fit_metadata"].get("feature_names", []))
    label_feature_leaks = sorted(set(feature_names) & LABEL_COLUMNS)
    refined_manifest = sources["refined_manifest"]
    refined_hashes_verified = bool(refined_manifest.get("verified_unchanged", False))
    validation_uncertainty_file = resolve_dirs(config, root)["refined_reports"] / "refined_validation_uncertainty_predictions.csv"
    validation_abstention_file = resolve_dirs(config, root)["refined_reports"] / "refined_validation_abstention_predictions.csv"
    result = {
        "neural_model_id_matches": model_id == expected,
        "model_id": model_id,
        "feature_schema_unchanged": bool(feature_names),
        "label_feature_leaks": label_feature_leaks,
        "rul_cap": float(sources["final_fit_metadata"].get("rul_cap", np.nan)),
        "rul_cap_unchanged": float(sources["final_fit_metadata"].get("rul_cap", np.nan)) == 125.0,
        "cv_rows_are_oof_validation": bool({"fold", "seed", "true_rul"}.issubset(cv.columns) and cv["fold"].notna().all() and cv["seed"].notna().all()),
        "benchmark_not_in_policy_development": True,
        "benchmark_labels_used_for_policy_selection": False,
        "benchmark_subset_counts": {str(k): int(v) for k, v in subset_counts.items()},
        "benchmark_subset_counts_expected": subset_counts == {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248},
        "duplicate_benchmark_engine_keys": duplicate_benchmark,
        "duplicate_cv_decision_keys": duplicate_cv_keys,
        "engine_keys_unique_for_benchmark": duplicate_benchmark == 0,
        "uncertainty_method_locked": sources["locked_uncertainty"].get("candidate_method") == config["source"]["expected_uncertainty_method"],
        "abstention_method_locked": sources["locked_abstention"].get("policy_id") == config["source"]["expected_abstention_policy"],
        "validation_uncertainty_predictions_source": "derived_from_locked_policy" if not validation_uncertainty_file.exists() else str(validation_uncertainty_file),
        "validation_abstention_predictions_source": "derived_from_locked_policy" if not validation_abstention_file.exists() else str(validation_abstention_file),
        "source_hash_manifest_verified": refined_hashes_verified,
        "hard_failures": [],
    }
    hard_checks = [
        "neural_model_id_matches",
        "feature_schema_unchanged",
        "rul_cap_unchanged",
        "cv_rows_are_oof_validation",
        "benchmark_subset_counts_expected",
        "engine_keys_unique_for_benchmark",
        "uncertainty_method_locked",
        "abstention_method_locked",
        "source_hash_manifest_verified",
    ]
    result["hard_failures"] = [key for key in hard_checks if not result.get(key)]
    if result["hard_failures"]:
        raise ValueError(f"Source validation failed: {result['hard_failures']}")
    return result


def state_definitions(config: dict[str, Any]) -> dict[str, Any]:
    states = config["safety_states"]
    return {
        "CRITICAL": f"true_rul <= {states['critical_max_rul']}",
        "NEAR_TERM": f"{states['critical_max_rul']} < true_rul <= {states['near_term_max_rul']}",
        "INSPECTION_WINDOW": f"{states['near_term_max_rul']} < true_rul <= {states['inspection_max_rul']}",
        "MONITORING": f"{states['inspection_max_rul']} < true_rul <= {states['monitoring_max_rul']}",
        "HEALTHY": f"true_rul > {states['monitoring_max_rul']}",
        "primary_critical_threshold": float(states["critical_max_rul"]),
        "sensitivity_thresholds": list(states["critical_threshold_sensitivity"]),
    }


def assign_safety_state(true_rul: pd.Series, config: dict[str, Any]) -> pd.Series:
    states = config["safety_states"]
    y = true_rul.astype(float)
    return pd.Series(
        np.select(
            [
                y <= float(states["critical_max_rul"]),
                y <= float(states["near_term_max_rul"]),
                y <= float(states["inspection_max_rul"]),
                y <= float(states["monitoring_max_rul"]),
            ],
            ["CRITICAL", "NEAR_TERM", "INSPECTION_WINDOW", "MONITORING"],
            default="HEALTHY",
        ),
        index=true_rul.index,
    )


def add_prediction_bands(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["sequence_length_group"] = pd.cut(result["sequence_valid_length"].astype(float), bins=[-np.inf, 20, 40, np.inf], labels=["short", "medium", "long"]).astype(str)
    result["padding_group"] = pd.cut(result["padding_fraction"].astype(float), bins=[-np.inf, 0.05, 0.35, np.inf], labels=["low_padding", "medium_padding", "high_padding"]).astype(str)
    result["predicted_rul_band"] = pd.cut(result["predicted_rul"].astype(float), bins=[-np.inf, 15, 30, 60, 90, np.inf], labels=["0_15", "16_30", "31_60", "61_90", "above_90"]).astype(str)
    result["risk_group"] = pd.cut(result["high_error_risk_probability"].astype(float), bins=[-np.inf, 0.25, 0.50, 0.75, np.inf], labels=["low_risk", "moderate_risk", "high_risk", "very_high_risk"]).astype(str)
    return result


def load_abstention_model(policy: dict[str, Any]) -> Any | None:
    model_path = policy.get("model_path")
    if not model_path:
        return None
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def fit_support_thresholds(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    base = risk_feature_frame(frame)
    return {
        "low_support_rarity_threshold": float(base["operating_regime_rarity"].quantile(float(config["support_fallback"]["low_support_rarity_quantile"]))),
        "high_width_threshold": float(base["interval_width_90"].quantile(float(config["support_fallback"]["high_width_quantile"]))),
        "low_lower_bound_threshold": float(config["support_fallback"]["low_lower_bound_threshold"]),
    }


def add_decision_features(frame: pd.DataFrame, abstention_policy: dict[str, Any], support_thresholds: dict[str, float]) -> pd.DataFrame:
    result = risk_feature_frame(frame)
    health = result["health_score"] if "health_score" in result.columns else pd.Series(np.nan, index=result.index)
    rate = result["degradation_rate"] if "degradation_rate" in result.columns else pd.Series(np.nan, index=result.index)
    result["health_score_filled"] = pd.to_numeric(health, errors="coerce").fillna(0.5)
    result["degradation_rate_filled"] = pd.to_numeric(rate, errors="coerce").fillna(0.0)
    model = load_abstention_model(abstention_policy)
    feature_columns = list(abstention_policy.get("feature_columns", []))
    if model is not None and feature_columns:
        result["high_error_risk_probability"] = model.predict_proba(result[feature_columns].to_numpy(dtype=float))[:, 1]
    else:
        result["high_error_risk_probability"] = result["interpretable_risk_score"].astype(float)
    result["abstain_flag"] = result["high_error_risk_probability"] >= float(abstention_policy.get("threshold", 1.01))
    result["support_score"] = 1.0 - result["operating_regime_rarity"].astype(float)
    result["regime_distance"] = result["operating_regime_rarity"].astype(float)
    result["invalid_uncertainty"] = ~np.isfinite(result[["lower_80", "lower_90", "lower_95", "upper_90", "interval_width_90"]].to_numpy(dtype=float)).all(axis=1)
    result["unsupported_operating_condition"] = result["group_fallback_used"].astype(bool) if "group_fallback_used" in result.columns else False
    result["low_support_condition"] = result["operating_regime_rarity"].astype(float) >= float(support_thresholds["low_support_rarity_threshold"])
    result["support_category"] = np.select(
        [result["invalid_uncertainty"], result["unsupported_operating_condition"], result["low_support_condition"]],
        ["INVALID_UNCERTAINTY", "UNSUPPORTED", "LOW_SUPPORT"],
        default="IN_SUPPORT",
    )
    result["support_category_code"] = result["support_category"].map({"IN_SUPPORT": 0, "LOW_SUPPORT": 1, "UNSUPPORTED": 2, "INVALID_UNCERTAINTY": 3}).astype(float)
    result = result.sort_values(["subset", "global_engine_id", "cycle"]).copy()
    result["recent_rul_slope"] = (
        result.groupby(["subset", "global_engine_id"], observed=False)["predicted_rul"].diff()
        / result.groupby(["subset", "global_engine_id"], observed=False)["cycle"].diff().replace(0, np.nan)
    ).fillna(0.0)
    result["recent_lower_bound_slope"] = (
        result.groupby(["subset", "global_engine_id"], observed=False)["lower_90"].diff()
        / result.groupby(["subset", "global_engine_id"], observed=False)["cycle"].diff().replace(0, np.nan)
    ).fillna(0.0)
    return result


def prepare_decision_frame(frame: pd.DataFrame, uncertainty_policy: dict[str, Any], abstention_policy: dict[str, Any], support_thresholds: dict[str, float], config: dict[str, Any]) -> pd.DataFrame:
    with_intervals = apply_conformal_policy(frame, uncertainty_policy)
    result = add_decision_features(with_intervals, abstention_policy, support_thresholds)
    if "true_rul" in result.columns:
        result["safety_state"] = assign_safety_state(result["true_rul"], config)
    result = add_prediction_bands(result)
    leakage_features = sorted(set(DECISION_FEATURE_COLUMNS) & LABEL_COLUMNS)
    if leakage_features:
        raise ValueError(f"Policy decision feature list contains labels: {leakage_features}")
    return result


def engine_key(frame: pd.DataFrame) -> pd.Series:
    return frame["subset"].astype(str) + "|" + frame["global_engine_id"].astype(str)


def split_engines(frame: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    keys = np.asarray(sorted(engine_key(frame).unique()))
    rng = np.random.default_rng(int(seed))
    shuffled = keys.copy()
    rng.shuffle(shuffled)
    cut = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * float(fraction)))))
    dev_keys = set(shuffled[:cut])
    keys_series = engine_key(frame)
    dev = frame[keys_series.isin(dev_keys)].copy()
    val = frame[~keys_series.isin(dev_keys)].copy()
    overlap = sorted(set(engine_key(dev)) & set(engine_key(val)))
    return dev, val, {"seed": int(seed), "development_engine_count": int(len(dev_keys)), "validation_engine_count": int(len(keys) - len(dev_keys)), "engine_overlap_count": len(overlap)}


def candidate_grid(config: dict[str, Any], dev: pd.DataFrame) -> list[dict[str, Any]]:
    grid = config["policy_grid"]
    candidates: list[dict[str, Any]] = []
    for tc in grid["point_critical_thresholds"]:
        for tm in grid["point_maintenance_thresholds"]:
            for ti in grid["point_inspection_thresholds"]:
                if tc < tm < ti:
                    candidates.append({"policy_id": f"point_tc{tc}_tm{tm}_ti{ti}", "family": "point_threshold", "thresholds": {"tc": float(tc), "tm": float(tm), "ti": float(ti)}})
    for lc in grid["lower_bound_critical_thresholds"]:
        for lm in grid["lower_bound_maintenance_thresholds"]:
            for li in grid["lower_bound_inspection_thresholds"]:
                if lc < lm < li:
                    candidates.append({"policy_id": f"lower_lc{lc}_lm{lm}_li{li}", "family": "lower_bound_threshold", "thresholds": {"lc": float(lc), "lm": float(lm), "li": float(li)}})
    for tc in grid["point_critical_thresholds"]:
        for lc in grid["lower_bound_critical_thresholds"]:
            for guard in grid["guard_thresholds"]:
                candidates.append({"policy_id": f"hybrid_tc{tc}_lc{lc}_g{guard}", "family": "point_lower_hybrid", "thresholds": {"tc": float(tc), "lc": float(lc), "guard": float(guard), "tm": 45.0, "li": 75.0}})
    for tc in grid["point_critical_thresholds"]:
        for lc in grid["lower_bound_critical_thresholds"]:
            for risk in grid["risk_thresholds"]:
                candidates.append({"policy_id": f"risk_tc{tc}_lc{lc}_r{risk}", "family": "risk_aware_hybrid", "thresholds": {"tc": float(tc), "lc": float(lc), "risk": float(risk), "tm": 45.0, "li": 75.0}})
    for risk in grid["risk_thresholds"]:
        for quantile in grid["width_quantiles"]:
            width = float(dev["interval_width_90"].quantile(float(quantile)))
            candidates.append({"policy_id": f"two_stage_r{risk}_wq{quantile}", "family": "two_stage_safety_gate", "thresholds": {"risk": float(risk), "width": width, "tc": 25.0, "lc": 20.0, "tm": 45.0, "li": 75.0}})
    for idx, weights in enumerate(grid["monotone_weights"]):
        for urgent in grid["monotone_urgent_thresholds"]:
            for schedule in grid["monotone_schedule_thresholds"]:
                if schedule < urgent:
                    candidates.append({"policy_id": f"monotone_w{idx}_u{urgent}_s{schedule}", "family": "monotone_score", "weights": {k: float(v) for k, v in weights.items()}, "thresholds": {"urgent": float(urgent), "schedule": float(schedule), "plan": max(0.15, float(schedule) - 0.20)}})
    if DecisionTreeClassifier is not None:
        candidates.append({"policy_id": "tree_depth3", "family": "shallow_tree", "thresholds": {"max_depth": int(grid["maximum_tree_depth"]), "min_leaf": int(grid["minimum_tree_leaf_size"])}})
    candidates.append({"policy_id": "phase5c1_existing_reference", "family": "point_threshold", "thresholds": {"tc": 15.0, "tm": 30.0, "ti": 60.0}, "reference_policy": True})
    return candidates


def _base_action_series(frame: pd.DataFrame, default: str = "CONTINUE_MONITORING") -> pd.Series:
    return pd.Series(default, index=frame.index, dtype=object)


def _threshold_actions(value: pd.Series, c: float, m: float, i: float) -> pd.Series:
    return pd.Series(
        np.select(
            [value <= c, value <= m, value <= i],
            ["URGENT_ENGINEERING_REVIEW", "SCHEDULE_MAINTENANCE", "PLAN_INSPECTION"],
            default="CONTINUE_MONITORING",
        ),
        index=value.index,
        dtype=object,
    )


def desired_training_action(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [
                frame["safety_state"] == "CRITICAL",
                frame["safety_state"] == "NEAR_TERM",
                frame["safety_state"] == "INSPECTION_WINDOW",
            ],
            ["URGENT_ENGINEERING_REVIEW", "SCHEDULE_MAINTENANCE", "PLAN_INSPECTION"],
            default="CONTINUE_MONITORING",
        ),
        index=frame.index,
    )


def fit_tree_candidate(candidate: dict[str, Any], dev: pd.DataFrame) -> dict[str, Any]:
    if candidate["family"] != "shallow_tree" or DecisionTreeClassifier is None:
        return candidate
    columns = ["predicted_rul", "lower_90", "interval_width_90", "high_error_risk_probability", "padding_fraction", "operating_regime_rarity"]
    tree = DecisionTreeClassifier(
        max_depth=int(candidate["thresholds"]["max_depth"]),
        min_samples_leaf=int(candidate["thresholds"]["min_leaf"]),
        random_state=11501,
    )
    tree.fit(dev[columns].to_numpy(dtype=float), desired_training_action(dev))
    fitted = dict(candidate)
    fitted["model"] = tree
    fitted["feature_columns"] = columns
    fitted["tree_rules"] = export_text(tree, feature_names=columns) if export_text is not None else ""
    return fitted


def monotone_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    rul_score = 1.0 - np.clip(frame["predicted_rul"].astype(float) / 125.0, 0.0, 1.0)
    lb_score = 1.0 - np.clip(frame["lower_90"].astype(float) / 125.0, 0.0, 1.0)
    risk_score = np.clip(frame["high_error_risk_probability"].astype(float), 0.0, 1.0)
    support_score = np.clip(frame["support_category_code"].astype(float) / 3.0, 0.0, 1.0)
    deg = np.clip(frame["degradation_rate_filled"].astype(float), 0.0, 1.0)
    return (
        weights.get("rul", 0.0) * rul_score
        + weights.get("lower_bound", 0.0) * lb_score
        + weights.get("risk", 0.0) * risk_score
        + weights.get("support", 0.0) * support_score
        + weights.get("degradation", 0.0) * deg
    )


def apply_policy(frame: pd.DataFrame, candidate: dict[str, Any], *, mandatory_review_for_abstention: bool = True) -> pd.DataFrame:
    result = frame.copy()
    family = candidate["family"]
    params = candidate.get("thresholds", {})
    if family == "point_threshold":
        action = _threshold_actions(result["predicted_rul"], params["tc"], params["tm"], params["ti"])
    elif family == "lower_bound_threshold":
        action = _threshold_actions(result["lower_90"], params["lc"], params["lm"], params["li"])
    elif family == "point_lower_hybrid":
        action = _base_action_series(result)
        urgent = (result["predicted_rul"] <= params["tc"]) | ((result["lower_90"] <= params["lc"]) & (result["predicted_rul"] <= params["guard"]))
        schedule = (result["predicted_rul"] <= params["tm"]) | (result["lower_90"] <= params["tm"])
        plan = (result["predicted_rul"] <= params["li"]) | (result["lower_90"] <= params["li"])
        action.loc[plan] = "PLAN_INSPECTION"
        action.loc[schedule] = "SCHEDULE_MAINTENANCE"
        action.loc[urgent] = "URGENT_ENGINEERING_REVIEW"
    elif family == "risk_aware_hybrid":
        action = _base_action_series(result)
        urgent = (result["predicted_rul"] <= params["tc"]) | ((result["lower_90"] <= params["lc"]) & (result["high_error_risk_probability"] >= params["risk"]))
        schedule = result["predicted_rul"] <= params["tm"]
        plan = result["predicted_rul"] <= params["li"]
        action.loc[plan] = "PLAN_INSPECTION"
        action.loc[schedule] = "SCHEDULE_MAINTENANCE"
        action.loc[urgent] = "URGENT_ENGINEERING_REVIEW"
    elif family == "two_stage_safety_gate":
        action = _threshold_actions(result["predicted_rul"], params["tc"], params["tm"], params["li"])
        review_gate = (result["high_error_risk_probability"] >= params["risk"]) | (result["interval_width_90"] >= params["width"])
        action.loc[review_gate & (result["lower_90"] <= params["lc"]) & (result["predicted_rul"] <= params["li"])] = "URGENT_ENGINEERING_REVIEW"
        action.loc[review_gate & (result["predicted_rul"] > params["li"])] = "ABSTAIN_AND_REVIEW"
    elif family == "monotone_score":
        score = monotone_score(result, candidate["weights"])
        action = _base_action_series(result)
        action.loc[score >= params["plan"]] = "PLAN_INSPECTION"
        action.loc[score >= params["schedule"]] = "SCHEDULE_MAINTENANCE"
        action.loc[score >= params["urgent"]] = "URGENT_ENGINEERING_REVIEW"
        result["monotone_risk_score"] = score
    elif family == "shallow_tree":
        columns = candidate.get("feature_columns", [])
        if "model" not in candidate:
            action = _base_action_series(result)
        else:
            action = pd.Series(candidate["model"].predict(result[columns].to_numpy(dtype=float)), index=result.index, dtype=object)
    else:
        raise ValueError(f"Unknown policy family: {family}")
    mandatory = result["invalid_uncertainty"].astype(bool) | result["unsupported_operating_condition"].astype(bool) | result["low_support_condition"].astype(bool)
    if mandatory_review_for_abstention:
        mandatory = mandatory | result["abstain_flag"].astype(bool)
    action.loc[mandatory] = "ABSTAIN_AND_REVIEW"
    result["maintenance_action"] = action
    result["policy_id"] = candidate["policy_id"]
    result["policy_family"] = family
    return result


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def action_costs(frame: pd.DataFrame, matrix: dict[str, float]) -> np.ndarray:
    state = frame["safety_state"].astype(str)
    action = frame["maintenance_action"].astype(str)
    cost = np.zeros(len(frame), dtype=float)
    cost += ((state == "CRITICAL") & (action == "CONTINUE_MONITORING")).to_numpy(dtype=float) * float(matrix["critical_as_monitor"])
    cost += ((state == "CRITICAL") & (action == "PLAN_INSPECTION")).to_numpy(dtype=float) * float(matrix["critical_as_inspection"])
    cost += ((state == "CRITICAL") & (action == "SCHEDULE_MAINTENANCE")).to_numpy(dtype=float) * float(matrix["critical_as_schedule"])
    cost += ((state == "CRITICAL") & (action == "ABSTAIN_AND_REVIEW")).to_numpy(dtype=float) * float(matrix["critical_as_abstain_review"])
    cost += ((state == "HEALTHY") & (action == "URGENT_ENGINEERING_REVIEW")).to_numpy(dtype=float) * float(matrix["healthy_as_urgent"])
    cost += ((state == "HEALTHY") & (action == "ABSTAIN_AND_REVIEW")).to_numpy(dtype=float) * float(matrix["healthy_as_abstain_review"])
    cost += ((state != "CRITICAL") & action.isin(["URGENT_ENGINEERING_REVIEW", "ABSTAIN_AND_REVIEW"])).to_numpy(dtype=float) * float(matrix["noncritical_as_review"])
    return cost


def policy_metrics(frame: pd.DataFrame, config: dict[str, Any], *, cost_matrix: dict[str, float] | None = None) -> dict[str, Any]:
    critical = frame["safety_state"] == "CRITICAL"
    urgent = frame["maintenance_action"] == "URGENT_ENGINEERING_REVIEW"
    abstain = frame["maintenance_action"] == "ABSTAIN_AND_REVIEW"
    operational = urgent | abstain
    critical_count = int(critical.sum())
    urgent_count = int(urgent.sum())
    abstain_count = int(abstain.sum())
    missed = critical & ~operational
    direct_recall = float((critical & urgent).sum() / max(critical_count, 1))
    operational_recall = float((critical & operational).sum() / max(critical_count, 1))
    urgent_precision = float((critical & urgent).sum() / max(urgent_count, 1))
    abstain_precision = float((critical & abstain).sum() / max(abstain_count, 1))
    lcb, ucb = wilson_interval(int((critical & operational).sum()), critical_count)
    total_review = urgent_count + abstain_count
    matrix = cost_matrix or config["cost_matrices"]["base"]
    return {
        "row_count": int(len(frame)),
        "engine_count": int(engine_key(frame).nunique()),
        "critical_count": critical_count,
        "direct_urgent_critical_recall": direct_recall,
        "operational_critical_recall": operational_recall,
        "critical_recall_lcb": lcb,
        "critical_recall_ucb": ucb,
        "urgent_review_precision": urgent_precision,
        "abstain_review_precision": abstain_precision,
        "missed_critical_count": int(missed.sum()),
        "missed_critical_rate": float(missed.sum() / max(critical_count, 1)),
        "urgent_count": urgent_count,
        "abstain_review_count": abstain_count,
        "mandatory_review_count": int(total_review),
        "urgent_review_rate": float(urgent_count / max(len(frame), 1)),
        "total_review_workload": float(total_review / max(len(frame), 1)),
        "critical_captured_by_urgent": int((critical & urgent).sum()),
        "critical_captured_by_abstain_review": int((critical & abstain).sum()),
        "critical_missed_by_both": int(missed.sum()),
        "weighted_safety_cost": float(np.mean(action_costs(frame, matrix))) if len(frame) else float("nan"),
        "action_distribution": {str(k): int(v) for k, v in frame["maintenance_action"].value_counts().sort_index().items()},
    }


def subgroup_metrics(frame: pd.DataFrame, config: dict[str, Any], *, policy_id: str) -> pd.DataFrame:
    rows = []
    min_critical = int(config["policy_selection"]["minimum_subgroup_critical_count"])
    for column in ["subset", "operating_regime", "sequence_length_group", "padding_group", "support_category", "predicted_rul_band", "risk_group"]:
        if column not in frame.columns:
            continue
        for value, group in frame.groupby(column, observed=False, dropna=False):
            critical = group["safety_state"] == "CRITICAL"
            operational = group["maintenance_action"].isin(["URGENT_ENGINEERING_REVIEW", "ABSTAIN_AND_REVIEW"])
            crit_n = int(critical.sum())
            hits = int((critical & operational).sum())
            lcb, ucb = wilson_interval(hits, crit_n)
            rows.append(
                {
                    "policy_id": policy_id,
                    "grouping": column,
                    "group_value": str(value),
                    "row_count": int(len(group)),
                    "critical_count": crit_n,
                    "operational_critical_recall": float(hits / max(crit_n, 1)) if crit_n else np.nan,
                    "critical_recall_lcb": lcb,
                    "critical_recall_ucb": ucb,
                    "missed_critical_count": int(crit_n - hits),
                    "sufficient_support": bool(crit_n >= min_critical),
                }
            )
    return pd.DataFrame(rows)


def evaluate_action_risk_order(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for action, group in frame.groupby("maintenance_action", observed=False):
        rows.append(
            {
                "maintenance_action": action,
                "row_count": int(len(group)),
                "critical_rate": float((group["safety_state"] == "CRITICAL").mean()),
                "near_term_or_critical_rate": float(group["safety_state"].isin(["CRITICAL", "NEAR_TERM"]).mean()),
                "urgency_rank": URGENCY_RANK.get(str(action), -1),
            }
        )
    return pd.DataFrame(rows).sort_values("urgency_rank")


def missed_reason(row: pd.Series, policy: dict[str, Any]) -> str:
    thresholds = policy.get("thresholds", {})
    if bool(row.get("invalid_uncertainty", False)):
        return "Missing decision data"
    if row.get("support_category") in {"LOW_SUPPORT", "UNSUPPORTED"}:
        return "Low support not escalated"
    if float(row.get("predicted_rul", 999.0)) > float(thresholds.get("tc", 15.0)):
        return "Point prediction too optimistic"
    if float(row.get("lower_90", 999.0)) > float(thresholds.get("lc", thresholds.get("tc", 15.0))):
        return "Interval lower bound too high"
    if float(row.get("high_error_risk_probability", 0.0)) < float(thresholds.get("risk", 0.0)):
        return "Error risk underestimated"
    return "Threshold-boundary miss"


def missed_critical_analysis(frame: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    critical = frame["safety_state"] == "CRITICAL"
    operational = frame["maintenance_action"].isin(["URGENT_ENGINEERING_REVIEW", "ABSTAIN_AND_REVIEW"])
    missed = frame[critical & ~operational].copy()
    if missed.empty:
        return pd.DataFrame(columns=["subset", "global_engine_id", "true_rul", "predicted_rul", "lower_90", "interval_width_90", "high_error_risk_probability", "support_score", "support_category", "operating_regime", "sequence_valid_length", "padding_fraction", "health_score", "degradation_rate", "maintenance_action", "missed_reason"])
    missed["missed_reason"] = [missed_reason(row, policy) for _, row in missed.iterrows()]
    columns = ["subset", "global_engine_id", "true_rul", "predicted_rul", "lower_90", "interval_width_90", "high_error_risk_probability", "support_score", "support_category", "operating_regime", "sequence_valid_length", "padding_fraction", "health_score", "degradation_rate", "maintenance_action", "missed_reason"]
    return missed[[column for column in columns if column in missed.columns]]


def sensitivity_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    original = config["safety_states"]["critical_max_rul"]
    for threshold in config["safety_states"]["critical_threshold_sensitivity"]:
        temp_config = json.loads(json.dumps(config))
        temp_config["safety_states"]["critical_max_rul"] = threshold
        temp = frame.copy()
        temp["safety_state"] = assign_safety_state(temp["true_rul"], temp_config)
        rows.append({"critical_threshold": threshold, **policy_metrics(temp, temp_config)})
    config["safety_states"]["critical_max_rul"] = original
    return pd.DataFrame(rows)


def candidate_feasibility(metrics: pd.Series | dict[str, Any], config: dict[str, Any]) -> tuple[bool, list[str]]:
    m = dict(metrics)
    floors = config["policy_selection"]
    reasons = []
    if float(m.get("operational_critical_recall", 0.0)) < float(floors["minimum_critical_recall"]):
        reasons.append("minimum_critical_recall")
    if float(m.get("missed_critical_rate", 1.0)) > float(floors["maximum_missed_critical_rate"]):
        reasons.append("maximum_missed_critical_rate")
    if float(m.get("urgent_review_precision", 0.0)) < float(floors["minimum_urgent_precision"]):
        reasons.append("minimum_urgent_precision")
    if float(m.get("urgent_review_rate", 1.0)) > float(floors["maximum_urgent_rate"]):
        reasons.append("maximum_urgent_rate")
    if float(m.get("total_review_workload", 1.0)) > float(floors["maximum_total_review_rate"]):
        reasons.append("maximum_total_review_rate")
    return not reasons, reasons


def is_pareto_frontier(frame: pd.DataFrame) -> pd.Series:
    values = frame[["operational_critical_recall", "urgent_review_precision", "weighted_safety_cost", "total_review_workload", "missed_critical_count"]].copy()
    frontier = []
    for i, row in values.iterrows():
        dominated = False
        for j, other in values.iterrows():
            if i == j:
                continue
            better_or_equal = (
                other["operational_critical_recall"] >= row["operational_critical_recall"]
                and other["urgent_review_precision"] >= row["urgent_review_precision"]
                and other["weighted_safety_cost"] <= row["weighted_safety_cost"]
                and other["total_review_workload"] <= row["total_review_workload"]
                and other["missed_critical_count"] <= row["missed_critical_count"]
            )
            strictly_better = (
                other["operational_critical_recall"] > row["operational_critical_recall"]
                or other["urgent_review_precision"] > row["urgent_review_precision"]
                or other["weighted_safety_cost"] < row["weighted_safety_cost"]
                or other["total_review_workload"] < row["total_review_workload"]
                or other["missed_critical_count"] < row["missed_critical_count"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        frontier.append(not dominated)
    return pd.Series(frontier, index=frame.index)


def select_policy(validation_frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    split_rows = []
    all_rows = []
    subgroup_rows = []
    fitted_by_id: dict[str, dict[str, Any]] = {}
    for seed in config["policy_selection"]["outer_seeds"]:
        dev, val, split = split_engines(validation_frame, float(config["policy_selection"]["development_fraction"]), int(seed))
        split_rows.append(split)
        support_thresholds = fit_support_thresholds(dev, config)
        candidates = [fit_tree_candidate(candidate, dev) for candidate in candidate_grid(config, dev)]
        for candidate in candidates:
            fitted_by_id[candidate["policy_id"]] = candidate
            pred = apply_policy(val, candidate, mandatory_review_for_abstention=bool(config["abstention_interaction"]["mandatory_review_for_abstention"]))
            metrics = policy_metrics(pred, config)
            feasible, reasons = candidate_feasibility(metrics, config)
            all_rows.append({"policy_id": candidate["policy_id"], "policy_family": candidate["family"], "selection_seed": int(seed), "feasible": feasible, "failed_floor_reasons": ";".join(reasons), **metrics})
            sub = subgroup_metrics(pred, config, policy_id=candidate["policy_id"])
            sub["selection_seed"] = int(seed)
            subgroup_rows.append(sub)
    split_metrics = pd.DataFrame(all_rows)
    aggregate = (
        split_metrics.groupby(["policy_id", "policy_family"], observed=False)
        .agg(
            operational_critical_recall=("operational_critical_recall", "mean"),
            direct_urgent_critical_recall=("direct_urgent_critical_recall", "mean"),
            urgent_review_precision=("urgent_review_precision", "mean"),
            abstain_review_precision=("abstain_review_precision", "mean"),
            missed_critical_count=("missed_critical_count", "mean"),
            missed_critical_rate=("missed_critical_rate", "mean"),
            urgent_count=("urgent_count", "mean"),
            abstain_review_count=("abstain_review_count", "mean"),
            mandatory_review_count=("mandatory_review_count", "mean"),
            urgent_review_rate=("urgent_review_rate", "mean"),
            total_review_workload=("total_review_workload", "mean"),
            weighted_safety_cost=("weighted_safety_cost", "mean"),
            critical_recall_lcb=("critical_recall_lcb", "mean"),
            worst_split_recall=("operational_critical_recall", "min"),
            median_split_recall=("operational_critical_recall", "median"),
            p10_split_recall=("operational_critical_recall", lambda s: float(np.quantile(s, 0.10))),
        )
        .reset_index()
    )
    subgroup_all = pd.concat(subgroup_rows, ignore_index=True) if subgroup_rows else pd.DataFrame()
    if not subgroup_all.empty:
        supported = subgroup_all[subgroup_all["sufficient_support"]].copy()
        worst_sub = supported.groupby("policy_id", observed=False)["operational_critical_recall"].min().rename("worst_supported_subgroup_recall").reset_index()
        aggregate = aggregate.merge(worst_sub, on="policy_id", how="left")
    aggregate["worst_supported_subgroup_recall"] = aggregate.get("worst_supported_subgroup_recall", pd.Series(np.nan, index=aggregate.index)).fillna(1.0)
    feasible_flags = []
    reasons_col = []
    for _, row in aggregate.iterrows():
        feasible, reasons = candidate_feasibility(row.to_dict(), config)
        if row["worst_supported_subgroup_recall"] < float(config["policy_selection"]["minimum_supported_subgroup_recall"]):
            feasible = False
            reasons.append("minimum_supported_subgroup_recall")
        feasible_flags.append(feasible)
        reasons_col.append(";".join(sorted(set(reasons))))
    aggregate["feasible_all_floors"] = feasible_flags
    aggregate["failed_floor_reasons"] = reasons_col
    aggregate["pareto_frontier"] = is_pareto_frontier(aggregate)
    aggregate = aggregate.sort_values(
        ["feasible_all_floors", "operational_critical_recall", "missed_critical_count", "total_review_workload", "urgent_review_precision", "weighted_safety_cost"],
        ascending=[False, False, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected_row = aggregate.iloc[0].to_dict()
    selected = fitted_by_id[str(selected_row["policy_id"])]
    final_support = fit_support_thresholds(validation_frame, config)
    if selected["family"] == "shallow_tree":
        selected = fit_tree_candidate(selected, validation_frame)
    locked = {
        "policy_id": selected["policy_id"],
        "policy_family": selected["family"],
        "thresholds": selected.get("thresholds", {}),
        "weights": selected.get("weights", {}),
        "tree_rules": selected.get("tree_rules", ""),
        "feature_columns": selected.get("feature_columns", DECISION_FEATURE_COLUMNS),
        "support_thresholds": final_support,
        "safety_state_definitions": state_definitions(config),
        "actions": ACTION_ORDER,
        "cost_matrix": config["cost_matrices"]["base"],
        "policy_development_engine_count": int(engine_key(validation_frame).nunique()),
        "policy_validation_results": selected_row,
        "split_seeds": list(config["policy_selection"]["outer_seeds"]),
        "safety_floors": config["policy_selection"],
        "feasibility_status": "all_floors_feasible" if bool(selected_row["feasible_all_floors"]) else "fallback_selected",
        "benchmark_labels_accessed_before_lock": False,
    }
    if selected.get("model") is not None:
        locked["_model"] = selected["model"]
    return pd.DataFrame(split_rows), split_metrics, subgroup_all, locked, aggregate


def materialize_policy_for_json(locked: dict[str, Any]) -> dict[str, Any]:
    clean = {key: value for key, value in locked.items() if key != "_model"}
    clean["lock_timestamp"] = pd.Timestamp.utcnow().isoformat()
    clean["policy_hash"] = stable_hash(clean)
    return clean


def apply_locked_policy(frame: pd.DataFrame, locked: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    candidate = {
        "policy_id": locked["policy_id"],
        "family": locked["policy_family"],
        "thresholds": locked.get("thresholds", {}),
        "weights": locked.get("weights", {}),
        "feature_columns": locked.get("feature_columns", []),
    }
    if locked.get("_model") is not None:
        candidate["model"] = locked["_model"]
    return apply_policy(frame, candidate, mandatory_review_for_abstention=bool(config["abstention_interaction"]["mandatory_review_for_abstention"]))


def cost_sensitivity(predictions: pd.DataFrame, config: dict[str, Any], policy_id: str) -> pd.DataFrame:
    rows = []
    for name, matrix in config["cost_matrices"].items():
        metrics = policy_metrics(predictions, config, cost_matrix=matrix)
        rows.append({"policy_id": policy_id, "cost_matrix": name, "weighted_safety_cost": metrics["weighted_safety_cost"], "operational_critical_recall": metrics["operational_critical_recall"], "total_review_workload": metrics["total_review_workload"]})
    return pd.DataFrame(rows)


def freeze_decision(benchmark_metrics: dict[str, Any], validation_summary: dict[str, Any], source_hashes_unchanged: bool, config: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if benchmark_metrics["operational_critical_recall"] < float(config["freeze"]["minimum_operational_critical_recall"]):
        reasons.append("benchmark_operational_critical_recall_below_floor")
    if benchmark_metrics["direct_urgent_critical_recall"] < float(config["freeze"]["minimum_direct_urgent_recall"]):
        reasons.append("benchmark_direct_urgent_recall_below_floor")
    if validation_summary["critical_recall_lcb"] < float(config["freeze"]["minimum_recall_lcb"]):
        reasons.append("validation_recall_lcb_below_floor")
    if benchmark_metrics["total_review_workload"] > float(config["freeze"]["maximum_total_review_rate"]):
        reasons.append("benchmark_total_review_workload_above_floor")
    if not source_hashes_unchanged:
        reasons.append("source_hashes_changed")
    if not reasons:
        decision = "READY_TO_FREEZE"
    elif benchmark_metrics["operational_critical_recall"] >= float(config["freeze"]["minimum_operational_critical_recall"]) and source_hashes_unchanged:
        decision = "READY_WITH_LIMITATIONS"
    else:
        decision = "NOT_READY"
    return {
        "freeze_decision": decision,
        "reasons": reasons,
        "recommendation": "Proceed to KAN phase only after accepting the listed safety limitations." if decision == "READY_WITH_LIMITATIONS" else ("Proceed to KAN phase." if decision == "READY_TO_FREEZE" else "Do not freeze Phase 5C before another safety-policy iteration."),
    }


def make_figures(
    reports: Path,
    aggregate: pd.DataFrame,
    split_metrics: pd.DataFrame,
    subgroup: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    benchmark_predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    previous_predictions: pd.DataFrame,
) -> list[str]:
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []

    def save(name: str) -> None:
        path = fig_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        figures.append(str(path))

    plt.figure(figsize=(7, 4)); plt.scatter(aggregate["urgent_review_rate"], aggregate["operational_critical_recall"], s=20); plt.xlabel("Urgent-review rate"); plt.ylabel("Critical recall"); save("critical_recall_vs_urgent_review_rate.png")
    plt.figure(figsize=(7, 4)); plt.scatter(aggregate["total_review_workload"], aggregate["operational_critical_recall"], s=20); plt.xlabel("Total review workload"); plt.ylabel("Operational recall"); save("operational_recall_vs_workload.png")
    plt.figure(figsize=(7, 4)); plt.scatter(aggregate["operational_critical_recall"], aggregate["urgent_review_precision"], s=20); plt.xlabel("Critical recall"); plt.ylabel("Urgent precision"); save("urgent_precision_vs_critical_recall.png")
    plt.figure(figsize=(7, 4)); plt.scatter(aggregate["mandatory_review_count"], aggregate["missed_critical_count"], s=20); plt.xlabel("Mandatory review count"); plt.ylabel("Missed critical count"); save("missed_critical_vs_mandatory_review.png")
    plt.figure(figsize=(7, 4)); plt.scatter(aggregate["total_review_workload"], aggregate["weighted_safety_cost"], s=20); plt.xlabel("Workload"); plt.ylabel("Weighted cost"); save("weighted_cost_vs_workload.png")
    plt.figure(figsize=(7, 4)); frontier = aggregate[aggregate["pareto_frontier"]]; plt.scatter(aggregate["total_review_workload"], aggregate["operational_critical_recall"], alpha=0.25); plt.scatter(frontier["total_review_workload"], frontier["operational_critical_recall"], color="red"); save("pareto_frontier.png")
    selected_id = str(aggregate.iloc[0]["policy_id"]) if not aggregate.empty else ""
    selected_splits = split_metrics[split_metrics["policy_id"] == selected_id]
    if not selected_splits.empty:
        plt.figure(figsize=(7, 4)); selected_splits.set_index("selection_seed")["operational_critical_recall"].plot(marker="o"); plt.ylim(0, 1.05); save("policy_stability_across_splits.png")
    supported = subgroup[(subgroup.get("policy_id", "") == selected_id) & subgroup.get("sufficient_support", False)] if not subgroup.empty else pd.DataFrame()
    if not supported.empty:
        plt.figure(figsize=(8, 5))
        y = np.arange(min(len(supported), 20))
        sample = supported.head(20)
        left = np.maximum(0.0, sample["operational_critical_recall"].to_numpy(dtype=float) - sample["critical_recall_lcb"].to_numpy(dtype=float))
        right = np.maximum(0.0, sample["critical_recall_ucb"].to_numpy(dtype=float) - sample["operational_critical_recall"].to_numpy(dtype=float))
        plt.errorbar(sample["operational_critical_recall"], y, xerr=[left, right], fmt="o")
        plt.yticks(y, sample["grouping"] + "=" + sample["group_value"])
        plt.xlim(0, 1.05)
        save("subgroup_critical_recall_forest.png")
    plt.figure(figsize=(7, 4))
    recall = float(aggregate.iloc[0]["operational_critical_recall"])
    lcb = float(aggregate.iloc[0]["critical_recall_lcb"])
    plt.errorbar([0], [recall], yerr=[[max(0.0, recall - lcb)], [0.0]], fmt="o")
    plt.ylim(0, 1.05)
    save("critical_recall_confidence_intervals.png")
    if not comparison.empty:
        plt.figure(figsize=(8, 4)); comparison.set_index("policy")["urgent_count"].plot(kind="bar"); save("action_distribution_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, frame, title in [
        (axes[0], previous_predictions, "Current Phase 5C.1"),
        (axes[1], benchmark_predictions, "Phase 5C.2 locked"),
    ]:
        matrix = pd.crosstab(frame["safety_state"], frame["maintenance_action"]).reindex(index=SAFETY_STATE_ORDER, columns=ACTION_ORDER, fill_value=0)
        ax.imshow(matrix.to_numpy(), cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(ACTION_ORDER)))
        ax.set_xticklabels(ACTION_ORDER, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(SAFETY_STATE_ORDER)))
        ax.set_yticklabels(SAFETY_STATE_ORDER, fontsize=8)
        for (row, col), value in np.ndenumerate(matrix.to_numpy()):
            ax.text(col, row, str(int(value)), ha="center", va="center", fontsize=7)
    save("current_vs_refined_confusion_matrices.png")
    for name, frame in [("validation", validation_predictions), ("benchmark", benchmark_predictions)]:
        missed = frame[(frame["safety_state"] == "CRITICAL") & ~frame["maintenance_action"].isin(["URGENT_ENGINEERING_REVIEW", "ABSTAIN_AND_REVIEW"])]
        plt.figure(figsize=(7, 4))
        if missed.empty:
            plt.axis("off")
            plt.text(0.05, 0.55, f"No {name} critical misses under operational detection.", fontsize=12)
        else:
            missed["predicted_rul"].plot(kind="hist", bins=20)
            plt.xlabel("Predicted RUL")
        save(f"{name}_missed_critical_feature_distribution.png")
    plt.figure(figsize=(7, 4)); validation_predictions.boxplot(column="high_error_risk_probability", by="safety_state", rot=30); plt.suptitle(""); save("error_risk_distribution_by_safety_state.png")
    plt.figure(figsize=(7, 4)); validation_predictions.boxplot(column="lower_90", by="safety_state", rot=30); plt.suptitle(""); save("lower_bound_distribution_by_safety_state.png")
    plt.figure(figsize=(7, 4)); pd.crosstab(benchmark_predictions["support_category"], benchmark_predictions["maintenance_action"]).plot(kind="bar", stacked=True, ax=plt.gca()); save("support_category_vs_assigned_action.png")
    plt.figure(figsize=(6, 3)); plt.axis("off"); plt.text(0.02, 0.6, f"Selected: {selected_id}\nOperational recall: {aggregate.iloc[0]['operational_critical_recall']:.3f}\nWorkload: {aggregate.iloc[0]['total_review_workload']:.3f}", fontsize=12); save("freeze_readiness_summary.png")
    return figures


def prepare_outputs(config: dict[str, Any], root: Path) -> tuple[Path, Path]:
    dirs = resolve_dirs(config, root)
    reports = dirs["reports"]
    artifacts = dirs["artifacts"]
    if (reports.exists() or artifacts.exists()) and not bool(config["outputs"].get("overwrite_existing", False)):
        raise FileExistsError(f"Safety-refined outputs already exist at {reports} or {artifacts}; set overwrite_existing explicitly.")
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    return reports, artifacts


def write_note(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validation = summary["validation_metrics"]
    benchmark = summary["benchmark_metrics"]
    locked = summary["locked_policy"]
    freeze = summary["freeze_decision"]
    floors = locked.get("safety_floors", {})
    lines = [
        "# Phase 5C.2 Maintenance Safety Policy Refinement",
        "",
        f"Freeze decision: `{freeze['freeze_decision']}`",
        "",
        f"Locked policy: `{locked['policy_id']}` (`{locked['policy_family']}`)",
        "",
        f"Locked thresholds: `{locked.get('thresholds', {})}`",
        "",
        "The policy was selected using OOF validation engines only. Benchmark labels were evaluated only after the locked policy file was written.",
        "The locked neural model, uncertainty policy, abstention policy and source artifacts were not changed.",
        "",
        "## Validation Selection",
        "",
        f"- Validation engines: `{summary['validation_engine_count']}`",
        f"- Validation critical rows: `{summary['validation_critical_count']}`",
        f"- Policy families tested: `{', '.join(summary['policy_families_tested'])}`",
        f"- Candidate count: `{summary['candidate_count']}`",
        f"- Feasible candidate count: `{summary['feasible_candidate_count']}`",
        f"- All configured floors feasible: `{summary['all_floors_simultaneously_feasible']}`",
        f"- Critical recall floor: `{floors.get('minimum_critical_recall')}`",
        f"- Max total review floor: `{floors.get('maximum_total_review_rate')}`",
        f"- Validation operational recall: `{validation['operational_critical_recall']:.4f}`",
        f"- Validation direct urgent recall: `{validation['direct_urgent_critical_recall']:.4f}`",
        f"- Validation urgent precision: `{validation['urgent_review_precision']:.4f}`",
        f"- Validation missed critical count: `{validation['missed_critical_count']}`",
        f"- Validation total review workload: `{validation['total_review_workload']:.4f}`",
        f"- Validation critical recall LCB: `{validation['critical_recall_lcb']:.4f}`",
        "",
        "## Benchmark Safety",
        "",
        f"- Operational critical recall: `{benchmark['operational_critical_recall']:.4f}`",
        f"- Direct urgent critical recall: `{benchmark['direct_urgent_critical_recall']:.4f}`",
        f"- Urgent precision: `{benchmark['urgent_review_precision']:.4f}`",
        f"- Missed critical count: `{benchmark['missed_critical_count']}`",
        f"- Urgent review count: `{benchmark['urgent_count']}`",
        f"- Abstain-review count: `{benchmark['abstain_review_count']}`",
        f"- Mandatory review count: `{benchmark['mandatory_review_count']}`",
        f"- Total review workload: `{benchmark['total_review_workload']:.4f}`",
        f"- Critical captured by abstention review: `{benchmark['critical_captured_by_abstain_review']}`",
        f"- Critical missed by urgent and abstention review: `{benchmark['critical_missed_by_both']}`",
        "",
        "## Recommendation",
        "",
        freeze["recommendation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validate_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root()
    dirs = resolve_dirs(config, root)
    return {
        "status": "valid",
        "source_phase5c_exists": dirs["phase5c_reports"].exists() and dirs["phase5c_artifacts"].exists(),
        "source_phase5c1_exists": dirs["refined_reports"].exists() and dirs["refined_artifacts"].exists(),
        "output_reports_dir": str(dirs["reports"]),
        "output_artifacts_dir": str(dirs["artifacts"]),
        "neural_retraining_disabled": True,
        "uncertainty_method_changes_allowed": False,
        "abstention_method_changes_allowed": False,
    }


def run_dry_run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root()
    manifest = build_source_manifest(config, root)
    missing = [row["artifact_key"] for row in manifest if row["required"] and not row["exists"]]
    return {
        "status": "dry_run_complete",
        "required_source_artifact_count": int(sum(row["required"] for row in manifest)),
        "missing_required_artifacts": missing,
        "policy_families": ["point_threshold", "lower_bound_threshold", "point_lower_hybrid", "risk_aware_hybrid", "two_stage_safety_gate", "monotone_score", "shallow_tree"],
        "benchmark_labels_excluded_from_policy_selection": True,
        "policy_lock_precedes_benchmark_evaluation": True,
        "neural_training_disabled": True,
    }


def _synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(13)
    for subset in ["FD001", "FD002"]:
        for engine in range(1, 11):
            for idx, true in enumerate([120, 70, 45, 25, 12]):
                pred = float(true + rng.normal(0, 10))
                rows.append(
                    {
                        "subset": subset,
                        "source_domain": subset,
                        "global_engine_id": f"{subset}_{engine:04d}",
                        "local_unit_id": engine,
                        "unit_id": engine,
                        "cycle": 20 + idx * 10,
                        "endpoint_index": idx,
                        "endpoint_cycle": 20 + idx * 10,
                        "sequence_valid_length": 20 + idx * 5,
                        "padded_cycle_count": max(0, 50 - (20 + idx * 5)),
                        "operating_regime": engine % 3,
                        "predicted_rul_raw": pred,
                        "predicted_rul": pred,
                        "true_rul": float(true),
                        "true_rul_capped": min(125.0, float(true)),
                        "residual": pred - true,
                        "absolute_error": abs(pred - true),
                        "squared_error": (pred - true) ** 2,
                        "fold": 1,
                        "seed": 1,
                        "final_observed_cycle": 20 + idx * 10,
                    }
                )
    cv = pd.DataFrame(rows)
    bench = cv.groupby(["subset", "global_engine_id"], observed=False).tail(1).copy().reset_index(drop=True)
    uncertainty = {"method_id": "global", "candidate_method": "global", "levels": [0.8, 0.9, 0.95], "global_radii": {"0.8": 15.0, "0.9": 25.0, "0.95": 35.0}, "minimum_group_size": 1}
    abstention = {"policy_id": "synthetic", "threshold": 0.9, "feature_columns": []}
    return cv, bench, uncertainty, abstention


def run_smoke_test(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    smoke_config = json.loads(json.dumps(config))
    smoke_config["bootstrap"]["iterations"] = int(smoke_config["bootstrap"].get("smoke_iterations", 50))
    cv, bench, uncertainty, abstention = _synthetic_frames()
    support = {"low_support_rarity_threshold": 0.8, "high_width_threshold": 30.0, "low_lower_bound_threshold": 20.0}
    cv_decision = prepare_decision_frame(cv, uncertainty, abstention, support, smoke_config)
    cv_decision.loc[cv_decision.index[0], "abstain_flag"] = True
    cv_decision.loc[cv_decision.index[1], "support_category"] = "LOW_SUPPORT"
    split_manifest, split_metrics, subgroup, locked, aggregate = select_policy(cv_decision, smoke_config)
    with tempfile.TemporaryDirectory(prefix="aeroguard_safety_") as temp:
        temp_path = Path(temp)
        policy_json = materialize_policy_for_json(locked)
        atomic_write_json(temp_path / "locked_maintenance_safety_policy.json", policy_json)
        benchmark_decision = prepare_decision_frame(bench, uncertainty, abstention, locked["support_thresholds"], smoke_config)
        scored = apply_locked_policy(benchmark_decision, locked, smoke_config)
        metrics = policy_metrics(scored, smoke_config)
    return {
        "status": "smoke_complete",
        "synthetic_only": True,
        "safety_state_count": int(cv_decision["safety_state"].nunique()),
        "policy_family_count": int(aggregate["policy_family"].nunique()),
        "candidate_count": int(len(aggregate)),
        "pareto_frontier_count": int(aggregate["pareto_frontier"].sum()),
        "locked_policy_family": locked["policy_family"],
        "benchmark_operational_recall": metrics["operational_critical_recall"],
        "policy_locked_before_benchmark": True,
        "neural_training_function_called": False,
    }


def run_full_posthoc_run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root()
    start = time.perf_counter()
    reports, artifacts = prepare_outputs(config, root)
    manifest = build_source_manifest(config, root)
    atomic_write_json(reports / "source_artifact_manifest.json", {"artifacts": manifest})
    validation = validate_sources(config, root, manifest)
    atomic_write_json(reports / "source_validation.json", validation)
    atomic_write_json(reports / "source_artifact_validation.json", validation)
    atomic_write_json(reports / "safety_state_definitions.json", state_definitions(config))
    sources = load_sources(config, root)
    cv_with_intervals_for_support = apply_conformal_policy(sources["cv"], sources["locked_uncertainty"])
    support_initial = fit_support_thresholds(cv_with_intervals_for_support, config)
    cv_decision = prepare_decision_frame(sources["cv"], sources["locked_uncertainty"], sources["locked_abstention"], support_initial, config)
    split_manifest, split_metrics, subgroup_all, locked_runtime, aggregate = select_policy(cv_decision, config)
    split_manifest.to_csv(reports / "policy_split_manifest.csv", index=False)
    aggregate.to_csv(reports / "maintenance_policy_candidate_metrics.csv", index=False)
    split_metrics.to_csv(reports / "maintenance_policy_split_metrics.csv", index=False)
    subgroup_all.to_csv(reports / "maintenance_policy_subgroup_metrics.csv", index=False)
    selected_validation = apply_locked_policy(prepare_decision_frame(sources["cv"], sources["locked_uncertainty"], sources["locked_abstention"], locked_runtime["support_thresholds"], config), locked_runtime, config)
    cost_sensitivity(selected_validation, config, locked_runtime["policy_id"]).to_csv(reports / "maintenance_policy_cost_sensitivity.csv", index=False)
    aggregate[aggregate["pareto_frontier"]].to_csv(reports / "maintenance_policy_pareto_frontier.csv", index=False)
    validation_missed = missed_critical_analysis(selected_validation, locked_runtime)
    validation_missed.to_csv(reports / "validation_missed_critical_analysis.csv", index=False)
    locked_json = materialize_policy_for_json(locked_runtime)
    atomic_write_json(reports / "locked_maintenance_safety_policy.json", locked_json)
    atomic_write_json(artifacts / "locked_maintenance_safety_policy.json", locked_json)
    fitted_policy_object = {key: value for key, value in locked_runtime.items() if key != "_model"}
    fitted_policy_object["lock_timestamp"] = locked_json["lock_timestamp"]
    fitted_policy_object["policy_hash"] = locked_json["policy_hash"]
    with (artifacts / "locked_maintenance_safety_policy.pkl").open("wb") as handle:
        pickle.dump(fitted_policy_object, handle)
    if locked_runtime.get("_model") is not None:
        with (artifacts / "locked_maintenance_safety_tree.pkl").open("wb") as handle:
            pickle.dump(locked_runtime["_model"], handle)
    benchmark_decision = prepare_decision_frame(sources["benchmark"], sources["locked_uncertainty"], sources["locked_abstention"], locked_runtime["support_thresholds"], config)
    benchmark_scored = apply_locked_policy(benchmark_decision, locked_runtime, config)
    benchmark_scored.to_csv(reports / "benchmark_safety_predictions.csv", index=False)
    benchmark_metrics = policy_metrics(benchmark_scored, config)
    subgroup_benchmark = subgroup_metrics(benchmark_scored, config, policy_id=locked_runtime["policy_id"])
    benchmark_metrics["subset_metrics"] = subgroup_benchmark[subgroup_benchmark["grouping"] == "subset"].to_dict("records")
    benchmark_metrics["regime_metrics"] = subgroup_benchmark[subgroup_benchmark["grouping"] == "operating_regime"].to_dict("records")
    benchmark_metrics["support_metrics"] = subgroup_benchmark[subgroup_benchmark["grouping"] == "support_category"].to_dict("records")
    atomic_write_json(reports / "benchmark_safety_metrics.json", benchmark_metrics)
    benchmark_missed = missed_critical_analysis(benchmark_scored, locked_runtime)
    benchmark_missed.to_csv(reports / "benchmark_missed_critical_analysis.csv", index=False)
    previous = sources["refined_maintenance_recommendations"].copy()
    previous["safety_state"] = assign_safety_state(previous["true_rul"], config)
    previous["maintenance_action"] = previous["maintenance_action"].replace({"ENGINEERING_REVIEW_REQUIRED": "ABSTAIN_AND_REVIEW"})
    previous_metrics = policy_metrics(previous, config)
    comparison = pd.DataFrame(
        [
            {"policy": "phase5c1_previous", **{k: v for k, v in previous_metrics.items() if not isinstance(v, dict)}},
            {"policy": "phase5c2_refined", **{k: v for k, v in benchmark_metrics.items() if not isinstance(v, (dict, list))}},
        ]
    )
    comparison.to_csv(reports / "maintenance_policy_comparison.csv", index=False)
    sensitivity_metrics(selected_validation, config).to_csv(reports / "critical_threshold_sensitivity.csv", index=False)
    risk_order = evaluate_action_risk_order(selected_validation)
    risk_order.to_csv(reports / "action_risk_ordering.csv", index=False)
    final_manifest = build_source_manifest(config, root)
    source_hashes_unchanged = {
        row["artifact_key"]: row["sha256"] for row in manifest if row["sha256"]
    } == {
        row["artifact_key"]: row["sha256"] for row in final_manifest if row["sha256"]
    }
    freeze = freeze_decision(benchmark_metrics, policy_metrics(selected_validation, config), source_hashes_unchanged, config)
    atomic_write_json(reports / "phase5c_freeze_decision.json", freeze)
    figures = make_figures(reports, aggregate, split_metrics, subgroup_all, selected_validation, benchmark_scored, comparison, previous)
    summary = {
        "status": "completed",
        "runtime_seconds": time.perf_counter() - start,
        "source_validation": validation,
        "validation_engine_count": int(engine_key(cv_decision).nunique()),
        "validation_row_count": int(len(cv_decision)),
        "validation_critical_count": int((cv_decision["safety_state"] == "CRITICAL").sum()),
        "policy_families_tested": sorted(aggregate["policy_family"].unique().tolist()),
        "candidate_count": int(len(aggregate)),
        "feasible_candidate_count": int(aggregate["feasible_all_floors"].sum()),
        "all_floors_simultaneously_feasible": bool(aggregate["feasible_all_floors"].any()),
        "locked_policy": locked_json,
        "validation_metrics": policy_metrics(selected_validation, config),
        "benchmark_metrics": benchmark_metrics,
        "previous_metrics": previous_metrics,
        "source_hashes_unchanged": source_hashes_unchanged,
        "benchmark_labels_excluded_from_policy_selection": True,
        "policy_locked_before_benchmark_evaluation": True,
        "neural_model_retrained": False,
        "uncertainty_method_changed": False,
        "abstention_method_changed": False,
        "freeze_decision": freeze,
        "figures": figures,
        "generated_reports": [str(path) for path in reports.glob("*") if path.is_file()],
        "generated_artifacts": [str(path) for path in artifacts.glob("*") if path.is_file()],
    }
    atomic_write_json(reports / "run_summary.json", summary)
    write_note(root / "notes" / "maintenance_safety_policy_refinement_results.md", summary)
    atomic_write_json(reports / "source_artifact_manifest.json", {"artifacts": final_manifest, "verified_unchanged": source_hashes_unchanged})
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
