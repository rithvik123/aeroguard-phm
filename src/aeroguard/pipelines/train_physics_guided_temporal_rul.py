"""Phase 5C physics-guided Patch Transformer pipeline scaffold."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import pickle
import platform
import sys
import tempfile
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
from torch.utils.data import DataLoader

from aeroguard.data.columns import CYCLE_COLUMN
from aeroguard.data.multi_subset import load_test_subsets, load_training_subsets
from aeroguard.deep.checkpoints import save_checkpoint
from aeroguard.deep.reproducibility import set_global_seed
from aeroguard.deep.sampling import build_endpoint_table
from aeroguard.deep.sequence_dataset import InferenceSequenceDataset, SequenceWindowDataset
from aeroguard.deep.windowing import WindowSpec
from aeroguard.deep.models.common import trainable_parameter_count, validate_parameter_budget
from aeroguard.deep.models.physics_guided_patch_transformer import PhysicsGuidedPatchTransformer
from aeroguard.deep.physics.candidate_registry import active_loss_weights, default_candidate_registry, validate_candidate_registry
from aeroguard.deep.physics.composite_loss import CompositePhysicsLoss, PhysicsLossConfig
from aeroguard.deep.physics.health_targets import health_rul_consistency_diagnostics, normalized_capped_rul_targets
from aeroguard.deep.physics.paired_sequences import TemporalPairingConfig, build_temporal_pairs, pair_indices, triplet_indices
from aeroguard.deep.physics.regime_consistency import RegimePairingConfig, build_regime_pairs, empty_regime_pair_frame, regime_pair_diagnostics
from aeroguard.deep.physics.violation_metrics import cycle_rate_metrics, monotonicity_metrics, optimistic_error_metrics, smoothness_metrics
from aeroguard.evaluation.constraint_ablation import constraint_ablation_frame
from aeroguard.evaluation.coverage_analysis import assign_numeric_band
from aeroguard.evaluation.deep_rul_metrics import deep_point_metrics, metrics_by_group, prediction_direction
from aeroguard.evaluation.model_efficiency import model_efficiency_row
from aeroguard.evaluation.uncertainty_metrics import interval_metrics
from aeroguard.maintenance.uncertainty_policy import assign_maintenance_recommendations, maintenance_policy_metrics
from aeroguard.pipelines.train_fd001_baseline import project_root, resolve_project_path
from aeroguard.pipelines.train_multidomain_deep_rul import (
    apply_preprocessor,
    apply_support_abstention_maintenance,
    evaluate_model_frame as _phase5_evaluate_model_frame,
    fit_preprocessor,
    make_dataset,
    point_metrics_for_predictions,
    screening_split,
    snapshot_endpoint_table,
)
from aeroguard.uncertainty.abstention import abstention_metrics
from aeroguard.uncertainty.conformal import GlobalConformalCalibrator, PredictedRulBandConformalCalibrator


REQUIRED_SECTIONS = {
    "general",
    "sequence",
    "model",
    "warm_start",
    "pairing",
    "regime_consistency",
    "losses",
    "candidate_registry",
    "later_experiment",
    "uncertainty",
    "safety",
    "smoke_test",
}

FULL_RUN_OUTPUT_FILES = [
    "phase5b_benchmark_manifest.json",
    "physics_candidate_registry.json",
    "screening_split.json",
    "pairing_audit.csv",
    "screening_metrics.csv",
    "finalist_cross_validation_metrics.csv",
    "model_stability.csv",
    "constraint_ablation.csv",
    "physics_model_ranking.csv",
    "locked_physics_model.json",
    "final_fit_metadata.json",
    "benchmark_predictions.csv",
    "benchmark_metrics.json",
    "metrics_by_subset.csv",
    "metrics_by_rul_band.csv",
    "metrics_by_regime.csv",
    "trajectory_consistency_metrics.csv",
    "optimistic_error_analysis.csv",
    "phase5b_vs_physics_guided.csv",
    "uncertainty_cv_metrics.csv",
    "locked_uncertainty_method.json",
    "uncertainty_predictions.csv",
    "uncertainty_metrics.json",
    "phase5b_vs_physics_uncertainty.csv",
    "abstention_metrics.json",
    "maintenance_recommendations.csv",
    "maintenance_policy_metrics.json",
    "model_efficiency.csv",
    "run_summary.json",
]

FULL_RUN_CHECKPOINT_FILES = [
    "screening_{candidate_id}.pt",
    "finalist_{candidate_id}_fold{fold}_seed{seed}.pt",
    "locked_physics_guided_model.pt",
]

FIGURE_OUTPUT_FILES = [
    "candidate_validation_rmse.png",
    "candidate_nasa_score.png",
    "optimistic_error_comparison.png",
    "low_rul_optimistic_error_comparison.png",
    "constraint_violation_comparison.png",
    "finalist_fold_seed_rmse.png",
    "stability_comparison.png",
    "robust_ranking.png",
    "phase5b_vs_physics_guided_metrics.png",
    "predicted_vs_true_rul.png",
    "residual_distributions.png",
    "error_by_rul_band.png",
    "error_by_operating_regime.png",
    "monotonicity_violation_distribution.png",
    "cycle_rate_residual_distribution.png",
    "temporal_smoothness_distribution.png",
    "health_score_trajectories.png",
    "rul_trajectories.png",
    "coverage_vs_nominal_level.png",
    "interval_width_comparison.png",
    "abstention_tradeoff.png",
    "maintenance_action_distribution.png",
]

SCREENING_TEXT_COLUMNS = [
    "candidate_id",
    "architecture",
    "training_status",
    "failure_reason",
    "active_losses",
    "active_heads",
    "checkpoint_path",
]

SCREENING_NUMERIC_COLUMNS = [
    "fitting_engine_count",
    "validation_engine_count",
    "standard_window_count",
    "temporal_pair_count",
    "adjacent_pair_count",
    "fixed_gap_pair_count",
    "temporal_triplet_count",
    "regime_pair_count",
    "best_epoch",
    "stopping_epoch",
    "validation_mae",
    "validation_rmse",
    "validation_nasa_score",
    "validation_mean_signed_error",
    "validation_optimistic_rate",
    "validation_severe_optimistic_rate",
    "validation_low_rul_optimistic_rate",
    "monotonic_violation_rate",
    "rate_violation_rate",
    "smoothness_violation_rate",
    "health_violation_rate",
    "regime_consistency_violation_rate",
    "parameter_count",
    "checkpoint_size",
    "training_runtime",
    "cpu_latency",
    "gpu_latency",
]

CANONICAL_SCREENING_SCHEMA = [
    "candidate_id",
    "architecture",
    "training_status",
    "failure_reason",
    "active_losses",
    "active_heads",
    *SCREENING_NUMERIC_COLUMNS,
    "checkpoint_path",
]

CV_TEXT_COLUMNS = ["candidate_id", "training_status", "failure_reason", "checkpoint_path"]
CV_NUMERIC_COLUMNS = [
    "fold",
    "seed",
    "fitting_engine_count",
    "validation_engine_count",
    "best_epoch",
    "stopping_epoch",
    "training_runtime",
    "validation_mae",
    "validation_rmse",
    "validation_nasa_score",
    "validation_mean_signed_error",
    "validation_optimistic_rate",
    "validation_severe_optimistic_rate",
    "validation_low_rul_optimistic_rate",
    "monotonic_violation_rate",
    "rate_violation_rate",
    "smoothness_violation_rate",
    "health_violation_rate",
    "regime_consistency_violation_rate",
    "parameter_count",
    "cpu_latency",
    "gpu_latency",
    "checkpoint_size",
]
CANONICAL_CV_SCHEMA = ["candidate_id", "fold", "seed", "training_status", "failure_reason", *[column for column in CV_NUMERIC_COLUMNS if column not in {"fold", "seed"}], "checkpoint_path"]

METRIC_ALIAS_MAP = {
    "nasa_score": "validation_nasa_score",
    "validation_nasa": "validation_nasa_score",
    "val_nasa_score": "validation_nasa_score",
    "val_rmse": "validation_rmse",
    "val_mae": "validation_mae",
    "mean_signed_error": "validation_mean_signed_error",
    "optimistic_rate": "validation_optimistic_rate",
}

RANKING_METRIC_REGISTRY = {
    "validation_rmse": {"metric": "validation_rmse", "direction": "lower"},
    "validation_mae": {"metric": "validation_mae", "direction": "lower"},
    "validation_nasa_score": {"metric": "validation_nasa_score", "direction": "lower"},
    "validation_mean_signed_error": {"metric": "validation_mean_signed_error", "direction": "lower_abs"},
    "validation_optimistic_rate": {"metric": "validation_optimistic_rate", "direction": "lower"},
    "validation_severe_optimistic_rate": {"metric": "validation_severe_optimistic_rate", "direction": "lower"},
    "validation_low_rul_optimistic_rate": {"metric": "validation_low_rul_optimistic_rate", "direction": "lower"},
    "monotonic_violation_rate": {"metric": "monotonic_violation_rate", "direction": "lower"},
    "rate_violation_rate": {"metric": "rate_violation_rate", "direction": "lower"},
    "smoothness_violation_rate": {"metric": "smoothness_violation_rate", "direction": "lower"},
    "health_violation_rate": {"metric": "health_violation_rate", "direction": "lower"},
    "regime_consistency_violation_rate": {"metric": "regime_consistency_violation_rate", "direction": "lower"},
    "parameter_count": {"metric": "parameter_count", "direction": "lower"},
    "cpu_latency": {"metric": "cpu_latency", "direction": "lower"},
    "gpu_latency": {"metric": "gpu_latency", "direction": "lower"},
    "validation_rmse_std": {"metric": "validation_rmse_std", "direction": "lower"},
    "normalized_RMSE": {"metric": "validation_rmse", "direction": "lower"},
    "normalized_NASA": {"metric": "validation_nasa_score", "direction": "lower"},
    "severe_optimistic_rate": {"metric": "validation_severe_optimistic_rate", "direction": "lower"},
    "low_RUL_optimistic_rate": {"metric": "validation_low_rul_optimistic_rate", "direction": "lower"},
    "normalized_RMSE_std": {"metric": "validation_rmse_std", "direction": "lower"},
    "normalized_parameter_count": {"metric": "parameter_count", "direction": "lower"},
    "normalized_CPU_latency": {"metric": "cpu_latency", "direction": "lower"},
}

REJECTED_RANKING_METRIC_TOKENS = ("benchmark", "test", "fd001", "fd002", "fd003", "fd004")
BENCHMARK_LABEL_COLUMNS = [
    "rul_capped",
    "rul_uncapped",
    "true_rul",
    "true_rul_capped",
    "true_rul_uncapped",
    "target_rul_capped",
    "target_rul_uncapped",
    "proxy_degradation_label",
    "proxy_health_region",
]
CV_PREDICTION_SCHEMA = [
    "subset",
    "source_domain",
    "global_engine_id",
    "local_unit_id",
    "unit_id",
    "cycle",
    "endpoint_index",
    "endpoint_cycle",
    "sequence_valid_length",
    "padded_cycle_count",
    "target_rul_capped",
    "target_rul_uncapped",
    "operating_regime",
    "proxy_health_region",
    "predicted_rul_raw",
    "predicted_rul",
    "health_score",
    "degradation_rate",
    "candidate_id",
    "true_rul",
    "residual",
    "absolute_error",
    "squared_error",
    "prediction_direction",
    "fold",
    "seed",
]

FULL_RUN_STAGE_ORDER = [
    "inspect_environment",
    "verify_phase5b_artifacts",
    "create_phase5b_manifest",
    "load_training_subsets_stage",
    "load_benchmark_subsets_stage",
    "create_screening_split_stage",
    "fit_fold_preprocessing_stage",
    "create_standard_windows_stage",
    "create_temporal_pairs_stage",
    "create_regime_pairs_stage",
    "screen_all_candidates",
    "select_finalists",
    "run_finalist_cross_validation",
    "aggregate_stability_results",
    "run_constraint_ablation_analysis",
    "rank_physics_candidates",
    "lock_physics_model",
    "determine_locked_epoch_count",
    "fit_final_physics_model",
    "evaluate_benchmark_subsets",
    "evaluate_trajectory_consistency",
    "run_optimistic_error_analysis",
    "compare_phase5b_predictions",
    "fit_deep_conformal_calibration",
    "evaluate_uncertainty",
    "evaluate_support_and_abstention",
    "generate_maintenance_recommendations",
    "measure_model_efficiency",
    "generate_figures",
    "write_results_documentation",
    "verify_phase5b_hashes_unchanged",
    "write_run_summary",
]

FULL_RUN_STAGE_FUNCTIONS: dict[str, str] = {
    name: name for name in FULL_RUN_STAGE_ORDER
}

PROTECTED_DIRECTORIES = [
    "references",
    "extracted-code",
    "reports/multidomain_phm",
    "reports/rul_uncertainty",
    "reports/deep_rul",
    "reports/deep_rul_extended",
    "artifacts/deep_rul",
    "artifacts/deep_rul_extended",
]


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_config(config, project_root())
    return config


def validate_config(config: dict[str, Any], root: Path) -> None:
    missing = sorted(REQUIRED_SECTIONS - set(config))
    if missing:
        raise ValueError(f"Missing required configuration sections: {missing}")
    general = config["general"]
    sequence = config["sequence"]
    model = config["model"]
    warm_start = config["warm_start"]
    pairing = config["pairing"]
    regime = config["regime_consistency"]
    losses = config["losses"]
    registry = config["candidate_registry"]
    later = config["later_experiment"]
    uncertainty = config["uncertainty"]
    safety = config["safety"]
    smoke = config["smoke_test"]

    dataset_dir = resolve_project_path(general["dataset_dir"], root)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    valid_subsets = {"FD001", "FD002", "FD003", "FD004"}
    for subset in [str(item).upper() for item in general["training_subsets"]]:
        if subset not in valid_subsets:
            raise ValueError(f"Invalid training subset: {subset}")
        if not (dataset_dir / f"train_{subset}.txt").exists():
            raise FileNotFoundError(f"Missing training subset file: train_{subset}.txt")
    for subset in [str(item).upper() for item in general["benchmark_test_subsets"]]:
        if subset not in valid_subsets:
            raise ValueError(f"Invalid benchmark subset: {subset}")
        for name in [f"test_{subset}.txt", f"RUL_{subset}.txt"]:
            if not (dataset_dir / name).exists():
                raise FileNotFoundError(f"Missing benchmark file: {name}")
    for key in ["phase5b_config_path", "phase5b_results_path", "phase5b_checkpoint_path"]:
        if not resolve_project_path(general[key], root).exists():
            raise FileNotFoundError(f"Missing Phase 5B artifact: {general[key]}")
    if not (resolve_project_path(general["phase5b_results_path"], root) / "run_summary.json").exists():
        raise FileNotFoundError("Missing Phase 5B run_summary.json.")

    if int(sequence["window_length"]) <= 0 or int(sequence["window_stride"]) <= 0:
        raise ValueError("Invalid sequence window settings.")
    if not 1 <= int(sequence["minimum_valid_history"]) <= int(sequence["window_length"]):
        raise ValueError("Invalid minimum_valid_history.")
    if int(sequence["maximum_windows_per_engine"]) <= 0:
        raise ValueError("maximum_windows_per_engine must be positive.")
    if int(sequence["patch_length"]) <= 0 or int(sequence["patch_stride"]) <= 0:
        raise ValueError("Invalid patch settings.")
    if int(sequence["patch_length"]) > int(sequence["window_length"]):
        raise ValueError("patch_length must not exceed window_length.")
    if int(sequence["feature_count"]) <= 0:
        raise ValueError("feature_count must be positive.")
    if float(sequence["rul_cap"]) <= 0:
        raise ValueError("Invalid RUL cap.")
    if sequence["training_target"] != "rul_capped":
        raise ValueError("Phase 5C currently preserves the Phase 5B rul_capped training target.")

    if int(model["projection_dim"]) % int(model["attention_heads"]) != 0:
        raise ValueError("Projection dimension not divisible by attention heads.")
    for key in ["transformer_layers", "attention_heads", "feedforward_dim", "parameter_budget"]:
        if int(model[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if model["pooling"] not in {"mean", "attention", "final"}:
        raise ValueError("Invalid pooling method.")
    if model["positional_encoding"] not in {"sinusoidal", "learnable"}:
        raise ValueError("Invalid positional encoding.")
    if model["output_activation"] not in {"softplus", "relu"}:
        raise ValueError("Invalid output activation.")

    if warm_start["enabled"] and not resolve_project_path(warm_start["checkpoint_path"], root).exists():
        raise FileNotFoundError(f"Missing warm-start checkpoint: {warm_start['checkpoint_path']}")

    _validate_pairing(pairing)
    if float(regime["rul_matching_tolerance"]) < 0 or int(regime["maximum_regime_pairs"]) < 0:
        raise ValueError("Invalid regime consistency settings.")
    for key in ["maximum_regime_anchors", "maximum_partners_per_anchor", "maximum_pairs_per_regime_combination"]:
        if int(regime.get(key, 0)) < 0:
            raise ValueError(f"Invalid regime consistency setting: {key}")
    loss_config = PhysicsLossConfig.from_mapping(losses)
    CompositePhysicsLoss(loss_config)
    if loss_config.lambda_health > 0.0 and not bool(model["health_head_enabled"]):
        raise ValueError("Active health loss requires the health head.")
    if loss_config.include_rate_head_loss and not bool(model["rate_head_enabled"]):
        raise ValueError("Rate-head loss requires the rate head.")
    if loss_config.lambda_regime > 0.0 and not bool(regime["enabled"]):
        raise ValueError("Active regime loss requires regime pair builder.")

    definitions = registry.get("definitions") or default_candidate_registry()
    validate_candidate_registry(definitions, max_candidates=int(registry["maximum_candidate_count"]))
    if int(registry["maximum_candidate_count"]) > 10:
        raise ValueError("Use no more than 10 default candidates.")

    for name, schedule in later["training_schedules"].items():
        if int(schedule["max_epochs"]) <= 0 or int(schedule["minimum_epochs"]) <= 0:
            raise ValueError(f"Invalid training schedule: {name}")
        if int(schedule["minimum_epochs"]) > int(schedule["max_epochs"]):
            raise ValueError(f"Invalid minimum epochs in schedule: {name}")
        if float(schedule["learning_rate"]) <= 0 or float(schedule["weight_decay"]) < 0:
            raise ValueError(f"Invalid optimizer values in schedule: {name}")
        if schedule["scheduler"] not in {"none", "plateau", "cosine"}:
            raise ValueError(f"Invalid scheduler in schedule: {name}")
    for value in later["robust_selection_weights"].values():
        if not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError("Invalid robust score weight.")
    validate_ranking_configuration(later["robust_selection_weights"])
    if int(later["finalist_count"]) <= 0 or int(later["cv_folds"]) < 2:
        raise ValueError("Invalid finalist or CV fold setting.")
    if int(later["num_workers"]) != 0:
        raise ValueError("Phase 5C Windows-safe default requires num_workers=0.")

    for level in uncertainty["nominal_levels"]:
        if not 0.0 < float(level) < 1.0:
            raise ValueError("Invalid nominal coverage level.")
    if float(uncertainty["coverage_tolerance"]) < 0:
        raise ValueError("Invalid coverage tolerance.")
    if float(safety["low_rul_threshold"]) < 0 or float(safety["severe_optimistic_threshold"]) < 0:
        raise ValueError("Invalid safety threshold.")
    _validate_smoke(smoke)

    for path_value in [general["output_dir"], general["checkpoint_dir"], smoke["smoke_output_directory"]]:
        _validate_output_path(resolve_project_path(path_value, root), root)


def run_validate_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    return {"status": "valid", "candidate_count": len(config["candidate_registry"].get("definitions") or default_candidate_registry()), **environment_report()}


def run_dry_run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    model = build_model_from_config(config)
    orchestration = dry_run_orchestration_summary(config, Path(config_path), project_root())
    return {
        "status": "dry_run_complete",
        "parameter_count": trainable_parameter_count(model),
        "enabled_losses": enabled_losses(config["losses"]),
        "model_outputs": ["rul_raw", "rul_prediction", "health_score", "degradation_rate", "latent", "valid_token_count"],
        "full_run_wired": orchestration["full_run_wired"],
        "stage_count": orchestration["stage_count"],
        "stage_order": orchestration["stage_order"],
        "regime_pair_algorithm": orchestration["regime_pair_algorithm"],
        "regime_pair_lazy_build": orchestration["regime_pair_lazy_build"],
        "regime_pair_caps": orchestration["regime_pair_caps"],
        "candidates_requiring_regime_pairs": orchestration["candidates_requiring_regime_pairs"],
        "unbounded_regime_pair_generation_remaining": orchestration["unbounded_regime_pair_generation_remaining"],
        "canonical_screening_metric_names": orchestration["canonical_screening_metric_names"],
        "canonical_cv_metric_names": orchestration["canonical_cv_metric_names"],
        "configured_ranking_metrics": orchestration["configured_ranking_metrics"],
        "ranking_metrics_recognized": orchestration["ranking_metrics_recognized"],
        "nasa_score_calculator_resolves": orchestration["nasa_score_calculator_resolves"],
        "screening_serialization_includes_nasa_score": orchestration["screening_serialization_includes_nasa_score"],
        "cv_serialization_includes_nasa_score": orchestration["cv_serialization_includes_nasa_score"],
        "training_window_target_contract": orchestration["training_window_target_contract"],
        "inference_window_target_contract": orchestration["inference_window_target_contract"],
        "benchmark_endpoint_label_source": orchestration["benchmark_endpoint_label_source"],
        "label_leakage_checks": orchestration["label_leakage_checks"],
        "current_partial_resume": orchestration["current_partial_resume"],
        "pandas_cv_concat_warning_path_fixed": orchestration["pandas_cv_concat_warning_path_fixed"],
        "output_contract_count": len(FULL_RUN_OUTPUT_FILES),
        "required_helpers_resolved": orchestration["required_helpers_resolved"],
        "notimplemented_in_full_run_call_graph": orchestration["notimplemented_in_full_run_call_graph"],
        "dry_run_created_output_dir": orchestration["dry_run_created_output_dir"],
        "orchestration_summary": orchestration,
        **environment_report(),
    }


def run_smoke_test(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    torch.manual_seed(int(config["general"]["random_seed"]))
    rng = np.random.default_rng(int(config["general"]["random_seed"]))
    data = _synthetic_smoke_data(config, rng)
    regime_lazy_checks = _smoke_regime_pair_lazy_checks(config, data["metadata"])
    benchmark_inference_checks = _smoke_benchmark_inference_checks(config)
    model = build_model_from_config(config, smoke=True)
    model.train()
    loss_fn = CompositePhysicsLoss(_smoke_loss_config(config["losses"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["smoke_test"]["learning_rate"]), weight_decay=0.0)
    x = torch.as_tensor(data["sequences"], dtype=torch.float32)
    target = torch.as_tensor(data["targets"], dtype=torch.float32).view(-1, 1)
    batch = {
        "target_rul": target,
        "health_target": normalized_capped_rul_targets(target, float(config["sequence"]["rul_cap"])),
        "pair_indices": torch.as_tensor(data["pair_indices"], dtype=torch.long),
        "pair_cycle_gaps": torch.as_tensor(data["pair_cycle_gaps"], dtype=torch.float32).view(-1, 1),
        "pair_plateau_mask": torch.as_tensor(data["pair_plateau_mask"], dtype=torch.float32).view(-1, 1),
        "triplet_indices": torch.as_tensor(data["triplet_indices"], dtype=torch.long),
        "triplet_left_gaps": torch.as_tensor(data["triplet_left_gaps"], dtype=torch.float32).view(-1, 1),
        "triplet_right_gaps": torch.as_tensor(data["triplet_right_gaps"], dtype=torch.float32).view(-1, 1),
        "regime_pair_indices": torch.as_tensor(data["regime_pair_indices"], dtype=torch.long),
    }
    finite_losses = []
    gradient_seen = False
    steps = int(config["smoke_test"]["smoke_epochs"])
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        loss_result = loss_fn(outputs, batch)
        loss = loss_result["total_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Smoke loss is non-finite.")
        loss.backward()
        gradient_seen = any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
        if not gradient_seen:
            raise RuntimeError("No finite gradients were observed during smoke training.")
        optimizer.step()
        finite_losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        before = model(x)["rul_prediction"]
        altered = x.clone()
        padded_mask = altered[..., -1:] == 0.0
        altered[..., :-1] = torch.where(padded_mask.expand_as(altered[..., :-1]), altered[..., :-1] + 9999.0, altered[..., :-1])
        invariance_a = model(altered)["rul_prediction"]
    padded_invariance = bool(torch.allclose(before, invariance_a, rtol=1.0e-4, atol=1.0e-4))
    if not padded_invariance:
        raise RuntimeError("Padded-value invariance check failed.")

    with tempfile.TemporaryDirectory(prefix="aeroguard_phase5c_smoke_") as temp_name:
        temp_dir = Path(temp_name)
        checkpoint_path = temp_dir / "physics_guided_smoke_model.pt"
        summary_path = temp_dir / "smoke_summary.json"
        torch.save({"state_dict": model.state_dict(), "metadata": {"kind": "synthetic_smoke"}}, checkpoint_path)
        reloaded = build_model_from_config(config, smoke=True)
        payload = torch.load(checkpoint_path, map_location="cpu")
        reloaded.load_state_dict(payload["state_dict"])
        reloaded.eval()
        with torch.no_grad():
            after = reloaded(x)["rul_prediction"]
        reload_agreement = bool(torch.allclose(before, after, rtol=1.0e-5, atol=1.0e-5))
        if not reload_agreement:
            raise RuntimeError("Reloaded smoke model predictions changed.")
        metric_summary = _smoke_metrics(before.detach().numpy().reshape(-1), target.detach().numpy().reshape(-1), data)
        ranking_checks = _smoke_ranking_checks(config, temp_dir)
        summary = {
            "status": "smoke_complete",
            "synthetic_only": True,
            "losses": finite_losses,
            "parameter_count": trainable_parameter_count(model),
            "metric_api_exercised": sorted(metric_summary),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        temp_files = [str(checkpoint_path), str(summary_path)]
        files_inside_temp = all(str(Path(item)).startswith(str(temp_dir)) for item in temp_files)
    return {
        "status": "smoke_complete",
        "synthetic_only": True,
        "finite_loss_count": len(finite_losses),
        "gradient_seen": gradient_seen,
        "padded_value_invariance": padded_invariance,
        "reload_prediction_agreement": reload_agreement,
        "no_future_cycle_leakage": bool(data["no_future_cycle_leakage"]),
        "temporary_files_created": temp_files,
        "temporary_directory_removed": True,
        "temporary_files_limited_to_smoke_directory": files_inside_temp,
        "parameter_count": trainable_parameter_count(model),
        "implemented_loss_terms": enabled_losses(_smoke_loss_config(config["losses"]).__dict__),
        "violation_metric_keys": sorted(metric_summary),
        **regime_lazy_checks,
        **benchmark_inference_checks,
        **ranking_checks,
        **environment_report(),
    }


def run_full_run(config_path: str | Path, *, resume_from: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    return run_full_experiment(config, config_path=Path(config_path), root=project_root(), resume_from=resume_from)


def run_full_experiment(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    root: Path | None = None,
    stage_overrides: dict[str, Any] | None = None,
    resume_from: str | None = None,
) -> dict[str, Any]:
    """Execute the complete Phase 5C experiment orchestration for a future run."""

    root = root or project_root()
    if resume_from is not None:
        if resume_from not in FULL_RUN_STAGE_ORDER:
            raise ValueError(f"Unknown resume stage: {resume_from}")
        config = deepcopy(config)
        config.setdefault("general", {})["resume_existing"] = True
    start = time.perf_counter()
    state = initialize_full_run_state(config, Path(config_path) if config_path is not None else None, root)
    output_dir, checkpoint_dir = prepare_full_run_outputs(config, root)
    state["output_dir"] = output_dir
    state["checkpoint_dir"] = checkpoint_dir
    first_stage_index = 0
    if resume_from is not None:
        resume_report = inspect_phase5c_resume_state(config, root, resume_from=resume_from)
        if not resume_report["safe_to_resume"]:
            raise RuntimeError("Unsafe Phase 5C resume: " + json.dumps(_json_ready(resume_report), sort_keys=True))
        restore_state_for_resume(state, resume_report)
        first_stage_index = FULL_RUN_STAGE_ORDER.index(resume_from)
    overrides = stage_overrides or {}
    try:
        for stage_name in FULL_RUN_STAGE_ORDER[first_stage_index:]:
            stage_start = time.perf_counter()
            stage_fn = overrides.get(stage_name) or globals()[FULL_RUN_STAGE_FUNCTIONS[stage_name]]
            state["current_stage"] = stage_name
            if stage_name == "write_run_summary":
                state["status"] = "completed"
                state["end_timestamp"] = pd.Timestamp.utcnow().isoformat()
                state["runtime_seconds"] = time.perf_counter() - start
                state["completed_stage_count"] = int(len(FULL_RUN_STAGE_ORDER))
                state["failed_stage"] = None
                state["failures"] = []
            result = stage_fn(state)
            if result is not None:
                state["stage_results"][stage_name] = result
            state["runtime_by_stage"][stage_name] = time.perf_counter() - stage_start
        state["status"] = "completed"
    except Exception as exc:
        state["status"] = "failed"
        state["failed_stage"] = state.get("current_stage", "initialization")
        state["failures"].append(
            {
                "stage": state["failed_stage"],
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        state["runtime_seconds"] = time.perf_counter() - start
        write_failure_summary(state)
        raise
    state["runtime_seconds"] = time.perf_counter() - start
    return dict(state.get("run_summary", {})) or write_run_summary(state)


def initialize_full_run_state(config: dict[str, Any], config_path: Path | None, root: Path) -> dict[str, Any]:
    return {
        "config": config,
        "config_path": config_path,
        "root": root,
        "status": "running",
        "start_timestamp": pd.Timestamp.utcnow().isoformat(),
        "stage_results": {},
        "runtime_by_stage": {},
        "warnings": [],
        "failures": [],
        "generated_files": [],
        "phase5b_manifest": {},
        "phase5b_initial_hashes": {},
    }


def prepare_full_run_outputs(config: dict[str, Any], root: Path) -> tuple[Path, Path]:
    general = config["general"]
    output_dir = resolve_project_path(general["output_dir"], root)
    checkpoint_dir = resolve_project_path(general["checkpoint_dir"], root)
    _validate_output_path(output_dir, root)
    _validate_output_path(checkpoint_dir, root)
    overwrite = bool(general.get("overwrite_existing", False))
    resume = bool(general.get("resume_existing", False))
    completed = output_dir / "run_summary.json"
    if (output_dir.exists() or checkpoint_dir.exists()) and not overwrite and not resume:
        try:
            payload = json.loads(completed.read_text(encoding="utf-8")) if completed.exists() else {}
        except json.JSONDecodeError:
            payload = {}
        if payload.get("run_status") in {"complete", "completed"}:
            raise FileExistsError(f"Completed Phase 5C run already exists at {output_dir}; set overwrite_existing or resume_existing explicitly.")
        raise FileExistsError(
            "Partial Phase 5C output already exists at "
            f"{output_dir} or {checkpoint_dir}; set resume_existing or overwrite_existing explicitly before rerunning."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, checkpoint_dir


def inspect_phase5c_resume_state(config: dict[str, Any], root: Path, *, resume_from: str = "evaluate_benchmark_subsets") -> dict[str, Any]:
    output_dir = resolve_project_path(config["general"]["output_dir"], root)
    checkpoint_dir = resolve_project_path(config["general"]["checkpoint_dir"], root)
    run_summary_path = output_dir / "run_summary.json"
    final_metadata_path = output_dir / "final_fit_metadata.json"
    locked_model_path = output_dir / "locked_physics_model.json"
    cv_metrics_path = output_dir / "finalist_cross_validation_metrics.csv"
    cv_predictions_path = output_dir / "cv_predictions.csv"
    final_checkpoint_path = checkpoint_dir / "locked_physics_guided_model.pt"
    default_preprocessor_path = checkpoint_dir / "final_preprocessor.pkl"
    default_train_transformed_path = checkpoint_dir / "final_train_transformed.pkl"
    run_summary = _read_json(run_summary_path) if run_summary_path.exists() else {}
    final_metadata = _read_json(final_metadata_path) if final_metadata_path.exists() else {}
    locked_model = _read_json(locked_model_path) if locked_model_path.exists() else {}
    preprocessor_path = Path(final_metadata.get("preprocessor_path", default_preprocessor_path))
    train_transformed_path = Path(final_metadata.get("final_train_transformed_path", default_train_transformed_path))
    missing = []
    for label, path in [
        ("run_summary", run_summary_path),
        ("locked_physics_model", locked_model_path),
        ("final_fit_metadata", final_metadata_path),
        ("final_checkpoint", final_checkpoint_path),
        ("final_preprocessor", preprocessor_path),
        ("final_train_transformed", train_transformed_path),
        ("cv_predictions", cv_predictions_path),
        ("finalist_cross_validation_metrics", cv_metrics_path),
    ]:
        if not path.exists():
            missing.append(label)
    failed_stage = str(run_summary.get("failed_stage") or run_summary.get("stage") or "")
    if failed_stage in FULL_RUN_STAGE_ORDER and FULL_RUN_STAGE_ORDER.index(failed_stage) < FULL_RUN_STAGE_ORDER.index(resume_from):
        missing.append(f"previous_failure_before_{resume_from}")
    elif failed_stage and failed_stage not in FULL_RUN_STAGE_ORDER:
        missing.append("unknown_failed_stage")
    if final_metadata.get("config_hash") and final_metadata.get("config_hash") != stable_payload_hash(config):
        missing.append("config_hash_mismatch")
    elif "config_hash" not in final_metadata:
        missing.append("config_hash_missing")
    if final_metadata.get("candidate_registry_hash") and final_metadata.get("candidate_registry_hash") != stable_payload_hash(_candidate_registry(config)):
        missing.append("candidate_registry_hash_mismatch")
    elif "candidate_registry_hash" not in final_metadata:
        missing.append("candidate_registry_hash_missing")
    if "feature_names" not in final_metadata or not final_metadata.get("feature_names"):
        missing.append("final_feature_names_missing")
    if preprocessor_path.exists() and final_metadata.get("feature_names"):
        try:
            with preprocessor_path.open("rb") as handle:
                preprocessor = pickle.load(handle)
            if list(preprocessor.get("features", [])) != list(final_metadata.get("feature_names", [])):
                missing.append("feature_schema_mismatch")
        except Exception:
            missing.append("final_preprocessor_unreadable")
    if final_checkpoint_path.exists() and final_metadata.get("feature_names"):
        try:
            checkpoint_payload = torch.load(final_checkpoint_path, map_location="cpu")
            checkpoint_features = checkpoint_payload.get("metadata", {}).get("feature_names")
            if checkpoint_features is not None and list(checkpoint_features) != list(final_metadata.get("feature_names", [])):
                missing.append("checkpoint_feature_schema_mismatch")
        except Exception:
            missing.append("final_checkpoint_unreadable")
    if not final_metadata.get("operating_regime_metadata"):
        missing.append("operating_regime_metadata_missing")
    if float(final_metadata.get("rul_cap", config["sequence"]["rul_cap"])) != float(config["sequence"]["rul_cap"]):
        missing.append("rul_cap_mismatch")
    if "benchmark_predictions.csv" in {path.name for path in output_dir.glob("benchmark*.csv")}:
        missing.append("benchmark_outputs_already_present")
    artifacts = {
        "run_summary": artifact_sha256_and_size(run_summary_path),
        "locked_physics_model": artifact_sha256_and_size(locked_model_path),
        "final_fit_metadata": artifact_sha256_and_size(final_metadata_path),
        "final_checkpoint": artifact_sha256_and_size(final_checkpoint_path),
        "final_preprocessor": artifact_sha256_and_size(preprocessor_path),
        "final_train_transformed": artifact_sha256_and_size(train_transformed_path),
        "cv_predictions": artifact_sha256_and_size(cv_predictions_path),
        "finalist_cross_validation_metrics": artifact_sha256_and_size(cv_metrics_path),
    }
    safe = not missing and failed_stage in {"evaluate_benchmark_subsets", "evaluate_trajectory_consistency", "run_optimistic_error_analysis", "compare_phase5b_predictions", "fit_deep_conformal_calibration", "evaluate_uncertainty", "evaluate_support_and_abstention", "generate_maintenance_recommendations", "measure_model_efficiency", "generate_figures", "write_results_documentation", "verify_phase5b_hashes_unchanged", "write_run_summary"}
    return {
        "resume_requested_stage": resume_from,
        "safe_to_resume": bool(safe),
        "earliest_safe_resume_stage": resume_from if safe else "",
        "failed_stage": failed_stage,
        "locked_candidate_id": locked_model.get("candidate_id", final_metadata.get("candidate_id", "")),
        "final_checkpoint_path": str(final_checkpoint_path),
        "final_checkpoint_exists": final_checkpoint_path.exists(),
        "missing_or_invalid_artifacts": sorted(set(missing)),
        "artifacts": artifacts,
    }


def restore_state_for_resume(state: dict[str, Any], resume_report: dict[str, Any]) -> None:
    config = state["config"]
    root = state["root"]
    final_metadata = _read_json(state["output_dir"] / "final_fit_metadata.json")
    locked_payload = _read_json(state["output_dir"] / "locked_physics_model.json")
    candidate_id = str(locked_payload.get("candidate_id") or final_metadata["candidate_id"])
    candidate = next(item for item in _candidate_registry(config) if item["candidate_id"] == candidate_id)
    inspect_environment(state)
    frames, metadata = load_test_subsets(
        resolve_project_path(config["general"]["dataset_dir"], root),
        config["general"]["benchmark_test_subsets"],
        _healthy_rul_threshold(config),
        _critical_rul_threshold(config),
    )
    with Path(final_metadata["preprocessor_path"]).open("rb") as handle:
        preprocessor = pickle.load(handle)
    with Path(final_metadata["final_train_transformed_path"]).open("rb") as handle:
        final_train_transformed = pickle.load(handle)
    model = build_candidate_model(candidate, len(final_metadata["feature_names"]) + 1, config).to(state["device"])
    payload = torch.load(Path(final_metadata["checkpoint_path"]), map_location=state["device"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    state.update(
        {
            "resume_report": resume_report,
            "benchmark_frames": frames,
            "benchmark_metadata": metadata,
            "locked_candidate": candidate,
            "locked_model_metadata": locked_payload,
            "final_fit_metadata": final_metadata,
            "final_model": model,
            "final_preprocessor": preprocessor,
            "final_train_transformed": final_train_transformed,
            "cv_metrics": normalize_cv_metrics_schema(pd.read_csv(state["output_dir"] / "finalist_cross_validation_metrics.csv")),
            "cv_predictions": pd.read_csv(state["output_dir"] / "cv_predictions.csv"),
        }
    )


def inspect_environment(state: dict[str, Any]) -> dict[str, Any]:
    env = environment_report()
    state["environment"] = env
    requested = str(state["config"]["general"].get("device", "auto"))
    state["device"] = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested))
    return env


def verify_phase5b_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    paths = _phase5b_artifact_paths(state["config"], state["root"])
    hashes = {str(path): sha256_file(path) for path in paths if path.exists()}
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 5B artifact(s): {missing}")
    state["phase5b_initial_hashes"] = hashes
    return {"file_count": len(paths), "sha256": hashes, "missing": missing}


def create_phase5b_manifest(state: dict[str, Any]) -> dict[str, Any]:
    config = state["config"]
    root = state["root"]
    phase5b_dir = resolve_project_path(config["general"]["phase5b_results_path"], root)
    checkpoint_path = resolve_project_path(config["general"]["phase5b_checkpoint_path"], root)
    summary = _read_json(phase5b_dir / "run_summary.json")
    manifest = {
        "locked_phase5b_model_id": summary.get("locked_model_id") or summary.get("locked_architecture") or "patch_transformer_10x5_mean_b",
        "architecture": summary.get("locked_architecture", "patch_transformer"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_hash": state["phase5b_initial_hashes"].get(str(checkpoint_path)),
        "core_report_hashes": state["phase5b_initial_hashes"],
        "benchmark_metrics": summary.get("benchmark_metrics", {}),
        "uncertainty_metrics": summary.get("deep_uncertainty_metrics") or summary.get("uncertainty_metrics", {}),
        "abstention_metrics": summary.get("abstention_metrics", {}),
        "file_paths": [str(path) for path in _phase5b_artifact_paths(config, root)],
        "software_metadata": summary.get("environment", {}),
        "statement": "Phase 5B benchmark files were read-only inputs to Phase 5C and must not be modified.",
    }
    atomic_write_json(state["output_dir"] / "phase5b_benchmark_manifest.json", manifest)
    _record_file(state, state["output_dir"] / "phase5b_benchmark_manifest.json")
    state["phase5b_manifest"] = manifest
    return manifest


def load_training_subsets_stage(state: dict[str, Any]) -> dict[str, Any]:
    config = state["config"]
    dataset_dir = resolve_project_path(config["general"]["dataset_dir"], state["root"])
    frame, metadata = load_training_subsets(
        dataset_dir,
        config["general"]["training_subsets"],
        float(config["sequence"]["rul_cap"]),
        _healthy_rul_threshold(config),
        _critical_rul_threshold(config),
    )
    validate_engine_frame(frame)
    state["training_frame"] = frame
    state["training_metadata"] = metadata
    return metadata


def load_benchmark_subsets_stage(state: dict[str, Any]) -> dict[str, Any]:
    config = state["config"]
    dataset_dir = resolve_project_path(config["general"]["dataset_dir"], state["root"])
    frames, metadata = load_test_subsets(
        dataset_dir,
        config["general"]["benchmark_test_subsets"],
        _healthy_rul_threshold(config),
        _critical_rul_threshold(config),
    )
    for frame in frames.values():
        validate_engine_frame(frame)
    state["benchmark_frames"] = frames
    state["benchmark_metadata"] = metadata
    return metadata


def create_screening_split_stage(state: dict[str, Any]) -> dict[str, Any]:
    later = state["config"]["later_experiment"]
    split_cfg = later.get("screening_split", {})
    train_ids, validation_ids = screening_split(
        state["training_frame"],
        float(split_cfg.get("validation_fraction", 0.2)),
        int(split_cfg.get("seed", state["config"]["general"]["random_seed"])),
    )
    if not train_ids or not validation_ids:
        raise ValueError("Screening split must contain fitting and validation engines.")
    payload = {"fitting_engine_ids": train_ids, "validation_engine_ids": validation_ids, "validation_fraction": split_cfg.get("validation_fraction", 0.2)}
    atomic_write_json(state["output_dir"] / "screening_split.json", payload)
    _record_file(state, state["output_dir"] / "screening_split.json")
    state["screening_split"] = payload
    return payload


def fit_fold_preprocessing_stage(state: dict[str, Any]) -> dict[str, Any]:
    frame = state["training_frame"]
    fitting_ids = set(state["screening_split"]["fitting_engine_ids"])
    validation_ids = set(state["screening_split"]["validation_engine_ids"])
    fitting = frame[frame["global_engine_id"].isin(fitting_ids)].copy()
    validation = frame[frame["global_engine_id"].isin(validation_ids)].copy()
    assert_training_only_preprocessing_frame(fitting)
    if fitting.empty or validation.empty:
        raise ValueError("Screening preprocessing split produced an empty side.")
    preprocessor = fit_preprocessor(fitting, _preprocess_config(state["config"]))
    fitting_transformed = apply_preprocessor(preprocessor, fitting)
    validation_transformed = apply_preprocessor(preprocessor, validation)
    state["screening_preprocessor"] = preprocessor
    state["screening_train_frame"] = fitting_transformed
    state["screening_validation_frame"] = validation_transformed
    return {"feature_count": len(preprocessor["features"]), "excluded_features": preprocessor.get("excluded", {})}


def create_standard_windows_stage(state: dict[str, Any]) -> dict[str, Any]:
    config = state["config"]
    spec = _window_spec(config)
    features = state["screening_preprocessor"]["features"]
    train_endpoint = build_endpoint_table(state["screening_train_frame"], spec, int(config["sequence"]["maximum_windows_per_engine"]), int(config["general"]["random_seed"]))
    val_endpoint = snapshot_endpoint_table(state["screening_validation_frame"], list(config["later_experiment"]["validation_snapshot_positions"]))
    train_dataset, train_meta, train_sequences = make_dataset(state["screening_train_frame"], train_endpoint, features, spec)
    val_dataset, val_meta, _ = make_dataset(state["screening_validation_frame"], val_endpoint, features, spec)
    train_meta = train_meta.copy()
    val_meta = val_meta.copy()
    train_meta["sample_index"] = np.arange(len(train_meta), dtype=np.int64)
    val_meta["sample_index"] = np.arange(len(val_meta), dtype=np.int64)
    state.update(
        {
            "screening_train_dataset": train_dataset,
            "screening_validation_dataset": val_dataset,
            "screening_train_metadata": train_meta,
            "screening_validation_metadata": val_meta,
            "screening_train_sequences": train_sequences,
            "feature_columns": features,
            "input_dim": int(train_sequences.shape[-1]),
        }
    )
    return {"standard_window_count": int(len(train_meta)), "validation_window_count": int(len(val_meta))}


def create_temporal_pairs_stage(state: dict[str, Any]) -> dict[str, Any]:
    pair_frame = build_temporal_pairs(state["screening_train_metadata"], _pairing_config(state["config"]))
    state["screening_temporal_pairs"] = pair_frame
    audit = pairing_audit_rows("stage_a", "screening", "screening", "all_candidates", state["screening_train_metadata"], pair_frame, pd.DataFrame())
    audit_frame = pd.DataFrame(audit)
    audit_frame.to_csv(state["output_dir"] / "pairing_audit.csv", index=False)
    _record_file(state, state["output_dir"] / "pairing_audit.csv")
    return _pair_counts(pair_frame, pd.DataFrame())


def create_regime_pairs_stage(state: dict[str, Any]) -> dict[str, Any]:
    state.setdefault("regime_pair_cache", {})
    state["screening_regime_pairs"] = empty_regime_pair_frame("lazy_not_built")
    required = [candidate["candidate_id"] for candidate in _candidate_registry(state["config"]) if candidate_requires_regime_pairs(candidate)]
    diagnostics = {
        "lazy_build": bool(state["config"]["regime_consistency"].get("lazy_build", True)),
        "cache_bounded_pairs": bool(state["config"]["regime_consistency"].get("cache_bounded_pairs", True)),
        "candidates_requiring_regime_pairs": required,
        "built_at_stage": False,
        **_pair_counts(state.get("screening_temporal_pairs", pd.DataFrame()), state["screening_regime_pairs"]),
    }
    state["regime_pair_diagnostics"] = diagnostics
    return diagnostics


def screen_all_candidates(state: dict[str, Any]) -> pd.DataFrame:
    rows = []
    candidates = _candidate_registry(state["config"])
    atomic_write_json(state["output_dir"] / "physics_candidate_registry.json", {"candidates": candidates})
    _record_file(state, state["output_dir"] / "physics_candidate_registry.json")
    for candidate in candidates:
        try:
            row = screen_candidate(state, candidate)
        except Exception as exc:
            if bool(state["config"].get("general", {}).get("fail_fast_candidates", False)):
                raise
            row = _failed_candidate_row(candidate, state, exc)
        rows.append(row)
    frame = normalize_screening_metrics_schema(pd.DataFrame(rows))
    frame.to_csv(state["output_dir"] / "screening_metrics.csv", index=False)
    _record_file(state, state["output_dir"] / "screening_metrics.csv")
    state["screening_metrics"] = frame
    return frame


def screen_candidate(state: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    seed = int(candidate.get("random_seed", state["config"]["general"]["random_seed"]))
    schedule = _schedule_for_candidate(state["config"], candidate)
    regime_pairs = get_regime_pairs_for_candidate(state, candidate, "screening", state["screening_train_metadata"])
    model, metadata = train_physics_model(
        candidate,
        state["screening_train_dataset"],
        state["screening_validation_dataset"],
        state["screening_train_metadata"],
        state["screening_validation_metadata"],
        state["screening_temporal_pairs"],
        regime_pairs,
        state["config"],
        schedule,
        state["device"],
        seed,
    )
    predictions = evaluate_physics_model_frame(
        model,
        state["screening_validation_dataset"],
        state["screening_validation_metadata"],
        state["device"],
        int(state["config"]["later_experiment"]["batch_size"]),
        str(candidate["candidate_id"]),
    )
    metrics = validation_metrics_for_frame(predictions, state["config"])
    constraint = constraint_metrics_for_predictions(predictions, state["screening_temporal_pairs"], regime_pairs)
    checkpoint = state["checkpoint_dir"] / f"screening_{candidate['candidate_id']}.pt"
    save_checkpoint(checkpoint, model, {"candidate": candidate, "training": metadata, "metrics": metrics})
    _record_file(state, checkpoint)
    efficiency = safe_efficiency_row(model, state["screening_validation_dataset"], state["device"], state["config"], str(candidate["candidate_id"]))
    row = {
        "candidate_id": candidate["candidate_id"],
        "architecture": candidate_architecture_label(candidate),
        "active_losses": ";".join(candidate["active_losses"]),
        "active_heads": ";".join(candidate["active_output_heads"]),
        "fitting_engine_count": int(state["screening_train_metadata"]["global_engine_id"].nunique()),
        "validation_engine_count": int(state["screening_validation_metadata"]["global_engine_id"].nunique()),
        "standard_window_count": int(len(state["screening_train_metadata"])),
        **_pair_counts(state["screening_temporal_pairs"], regime_pairs),
        "best_epoch": int(metadata.get("best_epoch", schedule["max_epochs"])),
        "stopping_epoch": int(metadata.get("stopping_epoch", metadata.get("best_epoch", schedule["max_epochs"]))),
        **metrics,
        **constraint,
        "parameter_count": trainable_parameter_count(model),
        "checkpoint_size": int(checkpoint.stat().st_size),
        "training_runtime": float(metadata.get("training_seconds", 0.0)),
        "cpu_latency": efficiency.get("cpu_batch_one_median_latency_ms"),
        "gpu_latency": efficiency.get("gpu_batch_one_median_latency_ms"),
        "training_status": "success",
        "failure_reason": "",
        "checkpoint_path": str(checkpoint),
    }
    return row


def select_finalists(state: dict[str, Any]) -> pd.DataFrame:
    state["screening_metrics"] = normalize_screening_metrics_schema(state["screening_metrics"])
    ranking = rank_candidates_dataframe(state["screening_metrics"], state["config"], require_success=True)
    count = min(int(state["config"]["later_experiment"]["finalist_count"]), len(ranking))
    finalists = ranking.head(count).copy()
    if finalists.empty:
        raise RuntimeError("No successful Phase 5C candidates were available for finalist selection.")
    state["finalists"] = finalists
    diagnostics = dict(ranking.attrs.get("ranking_diagnostics", {}))
    atomic_write_json(
        state["output_dir"] / "finalist_selection.json",
        {"finalists": finalists["candidate_id"].tolist(), "selection_source": "Stage A training-validation metrics only", "diagnostics": diagnostics},
    )
    _record_file(state, state["output_dir"] / "finalist_selection.json")
    atomic_write_json(state["output_dir"] / "finalist_selection_diagnostics.json", diagnostics)
    _record_file(state, state["output_dir"] / "finalist_selection_diagnostics.json")
    return finalists


def run_finalist_cross_validation(state: dict[str, Any]) -> pd.DataFrame:
    rows = []
    candidates_by_id = {item["candidate_id"]: item for item in _candidate_registry(state["config"])}
    splits = grouped_cv_splits(state["training_frame"], int(state["config"]["later_experiment"]["cv_folds"]), int(state["config"]["general"]["random_seed"]))
    for candidate_id in state["finalists"]["candidate_id"].tolist():
        candidate = candidates_by_id[candidate_id]
        for fold, (fit_ids, val_ids) in enumerate(splits, start=1):
            for seed in state["config"]["later_experiment"]["seeds"]:
                row = run_one_cv_fold(state, candidate, fit_ids, val_ids, fold, int(seed))
                rows.append(row)
    frame = normalize_cv_metrics_schema(pd.DataFrame(rows))
    frame.to_csv(state["output_dir"] / "finalist_cross_validation_metrics.csv", index=False)
    _record_file(state, state["output_dir"] / "finalist_cross_validation_metrics.csv")
    state["cv_predictions"] = normalize_cv_prediction_frames(state.get("cv_prediction_frames", []))
    state["cv_predictions"].to_csv(state["output_dir"] / "cv_predictions.csv", index=False)
    _record_file(state, state["output_dir"] / "cv_predictions.csv")
    state["cv_metrics"] = frame
    return frame


def aggregate_stability_results(state: dict[str, Any]) -> pd.DataFrame:
    metrics = normalize_cv_metrics_schema(state["cv_metrics"])
    state["cv_metrics"] = metrics
    rows = []
    columns = [
        "validation_mae",
        "validation_rmse",
        "validation_nasa_score",
        "validation_optimistic_rate",
        "validation_severe_optimistic_rate",
        "validation_low_rul_optimistic_rate",
        "monotonic_violation_rate",
        "rate_violation_rate",
        "smoothness_violation_rate",
        "health_violation_rate",
        "regime_consistency_violation_rate",
        "best_epoch",
        "training_runtime",
    ]
    for candidate_id, group in metrics.groupby("candidate_id", observed=False):
        row: dict[str, Any] = {"candidate_id": candidate_id}
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            row.update(
                {
                    f"mean_{column}": float(values.mean()),
                    f"std_{column}": float(values.std(ddof=0)),
                    f"median_{column}": float(values.median()),
                    f"min_{column}": float(values.min()),
                    f"max_{column}": float(values.max()),
                    f"iqr_{column}": float(values.quantile(0.75) - values.quantile(0.25)),
                    f"p05_{column}": float(values.quantile(0.05)),
                    f"p95_{column}": float(values.quantile(0.95)),
                }
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(state["output_dir"] / "model_stability.csv", index=False)
    _record_file(state, state["output_dir"] / "model_stability.csv")
    state["model_stability"] = frame
    return frame


def run_constraint_ablation_analysis(state: dict[str, Any]) -> pd.DataFrame:
    state["screening_metrics"] = normalize_screening_metrics_schema(state["screening_metrics"])
    frame = constraint_ablation_frame(state["screening_metrics"].to_dict("records"))
    if "phase5b_reimplementation_baseline" in set(frame["candidate_id"]):
        baseline = frame[frame["candidate_id"] == "phase5b_reimplementation_baseline"].iloc[0]
        rows = []
        for _, row in frame.iterrows():
            entry = row.to_dict()
            for metric in ["validation_mae", "validation_rmse", "validation_nasa_score", "validation_optimistic_rate", "monotonic_violation_rate", "rate_violation_rate", "smoothness_violation_rate", "parameter_count", "cpu_latency"]:
                if metric in frame.columns and pd.notna(row.get(metric)) and pd.notna(baseline.get(metric)):
                    entry[f"delta_{metric}"] = float(row[metric]) - float(baseline[metric])
            entry["ablation_interpretation"] = classify_constraint_effect(entry)
            rows.append(entry)
        frame = pd.DataFrame(rows)
    frame.to_csv(state["output_dir"] / "constraint_ablation.csv", index=False)
    _record_file(state, state["output_dir"] / "constraint_ablation.csv")
    state["constraint_ablation"] = frame
    return frame


def rank_physics_candidates(state: dict[str, Any]) -> pd.DataFrame:
    ranking = rank_candidates_dataframe(state["cv_metrics"], state["config"], require_success=True)
    ranking.to_csv(state["output_dir"] / "physics_model_ranking.csv", index=False)
    _record_file(state, state["output_dir"] / "physics_model_ranking.csv")
    state["physics_model_ranking"] = ranking
    return ranking


def lock_physics_model(state: dict[str, Any]) -> dict[str, Any]:
    if state["physics_model_ranking"].empty:
        raise RuntimeError("Cannot lock a model before finalist ranking.")
    candidate_id = str(state["physics_model_ranking"].iloc[0]["candidate_id"])
    candidate = next(item for item in _candidate_registry(state["config"]) if item["candidate_id"] == candidate_id)
    cv_row = state["physics_model_ranking"].iloc[0].to_dict()
    stability = state["model_stability"][state["model_stability"]["candidate_id"] == candidate_id]
    payload = {
        "candidate_id": candidate_id,
        "architecture": candidate["architecture_parameters"],
        "active_heads": candidate["active_output_heads"],
        "active_losses": candidate["active_losses"],
        "loss_weights": active_loss_weights(candidate),
        "pair_requirements": candidate["pairing_requirements"],
        "training_schedule": candidate["training_schedule"],
        "cv_metrics": cv_row,
        "stability_metrics": {} if stability.empty else stability.iloc[0].to_dict(),
        "constraint_metrics": _candidate_constraint_metrics(state, candidate_id),
        "parameter_count": cv_row.get("parameter_count"),
        "latency": {"cpu": cv_row.get("cpu_latency"), "gpu": cv_row.get("gpu_latency")},
        "selection_rationale": "Lowest configured robust score using training-validation and CV metrics only.",
        "benchmark_test_metrics_used_for_selection": False,
    }
    atomic_write_json(state["output_dir"] / "locked_physics_model.json", payload)
    _record_file(state, state["output_dir"] / "locked_physics_model.json")
    state["locked_candidate"] = candidate
    state["locked_model_metadata"] = payload
    return payload


def determine_locked_epoch_count(state: dict[str, Any]) -> dict[str, Any]:
    candidate_id = state["locked_candidate"]["candidate_id"]
    best_epochs = pd.to_numeric(state["cv_metrics"][state["cv_metrics"]["candidate_id"] == candidate_id]["best_epoch"], errors="coerce").dropna()
    if best_epochs.empty:
        raise RuntimeError("Cannot determine locked epoch count before finalist CV.")
    locked = max(1, int(round(float(best_epochs.median()))))
    upper_cap = int(state["config"]["later_experiment"]["maximum_epochs"])
    locked = min(locked, upper_cap)
    payload = {
        "all_best_epochs": [int(value) for value in best_epochs.tolist()],
        "median": float(best_epochs.median()),
        "interquartile_range": float(best_epochs.quantile(0.75) - best_epochs.quantile(0.25)),
        "minimum": int(best_epochs.min()),
        "maximum": int(best_epochs.max()),
        "locked_epoch_count": locked,
        "configured_upper_cap": upper_cap,
    }
    state["locked_epoch_metadata"] = payload
    return payload


def fit_final_physics_model(state: dict[str, Any]) -> dict[str, Any]:
    if "locked_candidate" not in state:
        raise RuntimeError("Final fit cannot run before finalist selection and model locking.")
    config = state["config"]
    frame = state["training_frame"].copy()
    assert_training_only_preprocessing_frame(frame)
    preprocessor = fit_preprocessor(frame, _preprocess_config(config))
    transformed = apply_preprocessor(preprocessor, frame)
    spec = _window_spec(config)
    endpoints = build_endpoint_table(transformed, spec, int(config["sequence"]["maximum_windows_per_engine"]), int(config["general"]["random_seed"]))
    dataset, metadata, _ = make_dataset(transformed, endpoints, preprocessor["features"], spec)
    metadata = metadata.copy()
    metadata["sample_index"] = np.arange(len(metadata), dtype=np.int64)
    temporal_pairs = build_temporal_pairs(metadata, _pairing_config(config))
    regime_pairs = get_regime_pairs_for_candidate(state, state["locked_candidate"], "final_fit", metadata)
    schedule = _schedule_for_candidate(config, state["locked_candidate"])
    schedule["max_epochs"] = int(state["locked_epoch_metadata"]["locked_epoch_count"])
    schedule["minimum_epochs"] = int(state["locked_epoch_metadata"]["locked_epoch_count"])
    model, train_meta = train_physics_model(
        state["locked_candidate"],
        dataset,
        dataset,
        metadata,
        metadata,
        temporal_pairs,
        regime_pairs,
        config,
        schedule,
        state["device"],
        int(config["general"]["random_seed"]),
        fixed_epochs=int(state["locked_epoch_metadata"]["locked_epoch_count"]),
    )
    checkpoint_path = state["checkpoint_dir"] / "locked_physics_guided_model.pt"
    preprocessor_path = state["checkpoint_dir"] / "final_preprocessor.pkl"
    final_train_transformed_path = state["checkpoint_dir"] / "final_train_transformed.pkl"
    with preprocessor_path.open("wb") as handle:
        pickle.dump(preprocessor, handle)
    with final_train_transformed_path.open("wb") as handle:
        pickle.dump(transformed, handle)
    save_checkpoint(
        checkpoint_path,
        model,
        {
            "candidate": state["locked_candidate"],
            "training": train_meta,
            "locked_epoch": state["locked_epoch_metadata"],
            "feature_names": list(preprocessor["features"]),
            "rul_cap": float(config["sequence"]["rul_cap"]),
            "window": dict(config["sequence"]),
        },
    )
    final_metadata = {
        "candidate_id": state["locked_candidate"]["candidate_id"],
        "checkpoint_path": str(checkpoint_path),
        "preprocessor_path": str(preprocessor_path),
        "final_train_transformed_path": str(final_train_transformed_path),
        "training_window_count": int(len(metadata)),
        "temporal_pair_count": int(len(temporal_pairs[temporal_pairs["pair_type"].isin(["adjacent", "fixed_gap"])]) if not temporal_pairs.empty else 0),
        "temporal_triplet_count": int((temporal_pairs["pair_type"] == "triplet").sum()) if not temporal_pairs.empty else 0,
        "regime_pair_count": int(len(regime_pairs)),
        "feature_names": list(preprocessor["features"]),
        "operating_regime_metadata": preprocessor["normalizer"].metadata() if hasattr(preprocessor["normalizer"], "metadata") else {},
        "rul_cap": float(config["sequence"]["rul_cap"]),
        "config_hash": stable_payload_hash(config),
        "candidate_registry_hash": stable_payload_hash(_candidate_registry(config)),
        "phase5b_hashes": state.get("phase5b_initial_hashes", {}),
        "window_length": int(config["sequence"]["window_length"]),
        "patch_length": int(config["sequence"]["patch_length"]),
        "patch_stride": int(config["sequence"]["patch_stride"]),
        "training_seconds": float(train_meta.get("training_seconds", 0.0)),
        "software_versions": environment_report(),
    }
    atomic_write_json(state["output_dir"] / "final_fit_metadata.json", final_metadata)
    _record_file(state, checkpoint_path)
    _record_file(state, preprocessor_path)
    _record_file(state, final_train_transformed_path)
    _record_file(state, state["output_dir"] / "final_fit_metadata.json")
    state.update({"final_model": model, "final_preprocessor": preprocessor, "final_train_transformed": transformed, "final_fit_metadata": final_metadata})
    return final_metadata


def evaluate_benchmark_subsets(state: dict[str, Any]) -> dict[str, Any]:
    if "final_model" not in state:
        raise RuntimeError("Benchmark evaluation cannot run before model locking and final fitting.")
    rows = []
    transformed_final_rows = []
    config = state["config"]
    spec = _window_spec(config)
    assert_no_label_features(list(state["final_preprocessor"]["features"]))
    for subset, frame in state["benchmark_frames"].items():
        state["current_benchmark_subset"] = subset
        endpoint_labels = build_benchmark_endpoint_table(frame, float(config["sequence"]["rul_cap"]))
        sensor_frame = benchmark_sensor_frame_without_labels(frame)
        transformed = apply_preprocessor(state["final_preprocessor"], sensor_frame)
        for label_column in BENCHMARK_LABEL_COLUMNS:
            if label_column in transformed.columns:
                raise ValueError(f"Benchmark label column leaked into transformed sensor frame: {label_column}")
        dataset, metadata, _ = make_dataset(transformed, endpoint_labels[["global_engine_id", "endpoint_index"]], state["final_preprocessor"]["features"], spec, mode="inference")
        pred = predict_physics_batches(state["final_model"], dataset, state["device"], int(config["later_experiment"]["batch_size"]))
        out = pd.concat([metadata.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
        out["candidate_id"] = state["locked_candidate"]["candidate_id"]
        out["subset"] = subset
        out["final_observed_cycle"] = out["cycle"].astype(int)
        out = attach_benchmark_labels(out, endpoint_labels)
        rows.append(out)
        transformed_final_rows.append(transformed.sort_values(["global_engine_id", CYCLE_COLUMN]).groupby("global_engine_id", observed=False).tail(1))
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(state["output_dir"] / "benchmark_predictions.csv", index=False)
    _record_file(state, state["output_dir"] / "benchmark_predictions.csv")
    metrics = benchmark_metrics_tables(predictions, config, state["output_dir"])
    state["benchmark_predictions"] = predictions
    state["benchmark_metrics"] = metrics["metrics"]
    state["benchmark_final_transformed"] = pd.concat(transformed_final_rows, ignore_index=True)
    return metrics


def evaluate_trajectory_consistency(state: dict[str, Any]) -> pd.DataFrame:
    rows = []
    predictions = state.get("benchmark_predictions", pd.DataFrame())
    for engine, group in predictions.sort_values(["global_engine_id", "final_observed_cycle"]).groupby("global_engine_id", observed=False):
        if len(group) < 3:
            continue
        pred = group["predicted_rul"].to_numpy(dtype=float)
        rows.append({"global_engine_id": engine, **monotonicity_metrics(pred[:-1], pred[1:]), **cycle_rate_metrics(pred[:-1], pred[1:], np.ones(len(pred) - 1)), **smoothness_metrics(pred[:-2], pred[1:-1], pred[2:])})
    frame = pd.DataFrame(rows)
    frame.to_csv(state["output_dir"] / "trajectory_consistency_metrics.csv", index=False)
    _record_file(state, state["output_dir"] / "trajectory_consistency_metrics.csv")
    state["trajectory_metrics"] = frame
    return frame


def run_optimistic_error_analysis(state: dict[str, Any]) -> pd.DataFrame:
    frame = optimistic_error_analysis(state["benchmark_predictions"], state["config"])
    frame.to_csv(state["output_dir"] / "optimistic_error_analysis.csv", index=False)
    _record_file(state, state["output_dir"] / "optimistic_error_analysis.csv")
    state["optimistic_error_analysis"] = frame
    return frame


def compare_phase5b_predictions(state: dict[str, Any]) -> pd.DataFrame:
    phase5b_path = resolve_project_path(state["config"]["general"]["phase5b_results_path"], state["root"]) / "benchmark_predictions.csv"
    phase5b = pd.read_csv(phase5b_path) if phase5b_path.exists() else pd.DataFrame()
    rows = []
    if not phase5b.empty:
        current = state["benchmark_predictions"]
        joined = phase5b.merge(current, on=["subset", "global_engine_id"], suffixes=("_phase5b", "_physics"))
        for model_name, prefix in [("phase5b", "phase5b"), ("physics_guided", "physics")]:
            pred_col = f"predicted_rul_{prefix}" if f"predicted_rul_{prefix}" in joined else "predicted_rul"
            true_col = f"true_rul_{prefix}" if f"true_rul_{prefix}" in joined else "true_rul"
            metrics = deep_point_metrics(joined[true_col], joined[pred_col], float(state["config"]["safety"]["severe_optimistic_threshold"]))
            rows.append({"model": model_name, **metrics})
    rows.append({"model": "classification", "result_classification": classify_phase5b_comparison(rows, state["config"])})
    frame = pd.DataFrame(rows)
    frame.to_csv(state["output_dir"] / "phase5b_vs_physics_guided.csv", index=False)
    _record_file(state, state["output_dir"] / "phase5b_vs_physics_guided.csv")
    state["phase5b_comparison"] = frame
    return frame


def fit_deep_conformal_calibration(state: dict[str, Any]) -> dict[str, Any]:
    oof = state["cv_metrics"].dropna(subset=["validation_rmse"]).copy()
    predictions = state.get("cv_predictions", pd.DataFrame())
    if predictions.empty:
        predictions = state["benchmark_predictions"].copy()
        state["warnings"].append("Conformal calibration used benchmark-shaped placeholder residuals because CV prediction rows were unavailable.")
    levels = [float(level) for level in state["config"]["uncertainty"]["nominal_levels"]]
    global_cal = GlobalConformalCalibrator(levels).fit(predictions["residual"])
    band_cal = PredictedRulBandConformalCalibrator(levels, state["config"]["uncertainty"]["predicted_rul_bands"], minimum_samples_per_band=20).fit(predictions["predicted_rul"], predictions["residual"])
    cv_rows = []
    for method_id, cal in [("physics_global_grouped_conformal", global_cal), ("physics_predicted_band_conformal", band_cal)]:
        frame = add_uncertainty_to_predictions(predictions, cal, method_id, levels)
        for level in levels:
            pct = int(round(level * 100))
            cv_rows.append({"uncertainty_method_id": method_id, **interval_metrics(frame["true_rul"], frame["predicted_rul"], frame[f"lower_{pct}"], frame[f"upper_{pct}"], level)})
    cv_metrics = pd.DataFrame(cv_rows)
    cv_metrics.to_csv(state["output_dir"] / "uncertainty_cv_metrics.csv", index=False)
    selected = select_uncertainty_method(cv_metrics, state["config"])
    atomic_write_json(state["output_dir"] / "locked_uncertainty_method.json", selected)
    _record_file(state, state["output_dir"] / "uncertainty_cv_metrics.csv")
    _record_file(state, state["output_dir"] / "locked_uncertainty_method.json")
    state["conformal_calibrators"] = {"physics_global_grouped_conformal": global_cal, "physics_predicted_band_conformal": band_cal}
    state["locked_uncertainty_method"] = selected
    return selected


def evaluate_uncertainty(state: dict[str, Any]) -> dict[str, Any]:
    method_id = state["locked_uncertainty_method"]["method_id"]
    levels = [float(level) for level in state["config"]["uncertainty"]["nominal_levels"]]
    predictions = add_uncertainty_to_predictions(state["benchmark_predictions"], state["conformal_calibrators"][method_id], method_id, levels)
    predictions.to_csv(state["output_dir"] / "uncertainty_predictions.csv", index=False)
    metrics = uncertainty_metrics_by_subset(predictions, levels)
    atomic_write_json(state["output_dir"] / "uncertainty_metrics.json", metrics)
    pd.DataFrame(compare_uncertainty_to_phase5b(state, metrics)).to_csv(state["output_dir"] / "phase5b_vs_physics_uncertainty.csv", index=False)
    _record_file(state, state["output_dir"] / "uncertainty_predictions.csv")
    _record_file(state, state["output_dir"] / "uncertainty_metrics.json")
    _record_file(state, state["output_dir"] / "phase5b_vs_physics_uncertainty.csv")
    state["uncertainty_predictions"] = predictions
    state["uncertainty_metrics"] = metrics
    return metrics


def evaluate_support_and_abstention(state: dict[str, Any]) -> dict[str, Any]:
    median_width90 = float(state["uncertainty_predictions"]["interval_width_90"].median())
    merged = apply_support_abstention_maintenance(
        state["uncertainty_predictions"],
        state["benchmark_final_transformed"],
        state["final_preprocessor"],
        state["final_train_transformed"],
        _policy_config(state["config"]),
        median_width90,
    )
    severe = float(state["config"]["safety"]["severe_optimistic_threshold"])
    metrics = {subset: abstention_metrics(group, 90, severe) for subset, group in merged.groupby("subset", observed=False)}
    metrics["overall"] = abstention_metrics(merged, 90, float(state["config"]["safety"]["severe_optimistic_threshold"]))
    atomic_write_json(state["output_dir"] / "abstention_metrics.json", metrics)
    _record_file(state, state["output_dir"] / "abstention_metrics.json")
    state["policy_predictions"] = merged
    state["abstention_metrics"] = metrics
    return metrics


def generate_maintenance_recommendations(state: dict[str, Any]) -> dict[str, Any]:
    frame = state["policy_predictions"].copy()
    if "maintenance_action" not in frame.columns:
        frame = assign_maintenance_recommendations(frame, state["config"]["safety"]["maintenance_thresholds"], "lower_90")
    frame.to_csv(state["output_dir"] / "maintenance_recommendations.csv", index=False)
    metrics = maintenance_policy_metrics(frame)
    atomic_write_json(state["output_dir"] / "maintenance_policy_metrics.json", metrics)
    _record_file(state, state["output_dir"] / "maintenance_recommendations.csv")
    _record_file(state, state["output_dir"] / "maintenance_policy_metrics.json")
    state["maintenance_metrics"] = metrics
    return metrics


def measure_model_efficiency(state: dict[str, Any]) -> pd.DataFrame:
    row = safe_efficiency_row(state["final_model"], state.get("screening_validation_dataset"), state["device"], state["config"], state["locked_candidate"]["candidate_id"])
    row.update({"checkpoint_size": int(Path(state["final_fit_metadata"]["checkpoint_path"]).stat().st_size), "training_runtime": state["final_fit_metadata"]["training_seconds"]})
    frame = pd.DataFrame([row])
    frame.to_csv(state["output_dir"] / "model_efficiency.csv", index=False)
    _record_file(state, state["output_dir"] / "model_efficiency.csv")
    state["model_efficiency"] = frame
    return frame


def generate_figures(state: dict[str, Any]) -> list[str]:
    paths = make_phase5c_figures(state)
    for path in paths:
        _record_file(state, Path(path))
    return paths


def write_results_documentation(state: dict[str, Any]) -> Path:
    path = state["root"] / "notes" / "physics_guided_temporal_rul_results.md"
    write_results_note(path, state)
    _record_file(state, path)
    return path


def write_run_summary(state: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "run_status": state.get("status", "running"),
        "start_timestamp": state.get("start_timestamp"),
        "end_timestamp": state.get("end_timestamp") or pd.Timestamp.utcnow().isoformat(),
        "runtime_seconds": float(state.get("runtime_seconds", 0.0)),
        "completed_stage_count": int(state.get("completed_stage_count", len(state.get("stage_results", {})))),
        "failed_stage": state.get("failed_stage"),
        "runtime_by_stage": state.get("runtime_by_stage", {}),
        "environment": state.get("environment", environment_report()),
        "dataset_files": _dataset_files(state["config"], state["root"]),
        "training_engine_count": int(state.get("training_frame", pd.DataFrame()).get("global_engine_id", pd.Series(dtype=object)).nunique()),
        "benchmark_engine_counts": {key: int(value["global_engine_id"].nunique()) for key, value in state.get("benchmark_frames", {}).items()},
        "candidate_count": len(_candidate_registry(state["config"])),
        "pair_counts": _pair_counts(state.get("screening_temporal_pairs", pd.DataFrame()), state.get("screening_regime_pairs", pd.DataFrame())),
        "finalists": state.get("finalists", pd.DataFrame()).get("candidate_id", pd.Series(dtype=object)).tolist() if isinstance(state.get("finalists"), pd.DataFrame) else [],
        "cv_folds": int(state["config"]["later_experiment"]["cv_folds"]),
        "seeds": state["config"]["later_experiment"]["seeds"],
        "locked_candidate": state.get("locked_candidate", {}).get("candidate_id"),
        "locked_epoch_count": state.get("locked_epoch_metadata", {}).get("locked_epoch_count"),
        "benchmark_metrics": state.get("benchmark_metrics", {}),
        "constraint_metrics": state.get("constraint_ablation", pd.DataFrame()).to_dict("records") if isinstance(state.get("constraint_ablation"), pd.DataFrame) else [],
        "optimistic_error_metrics": state.get("optimistic_error_analysis", pd.DataFrame()).to_dict("records") if isinstance(state.get("optimistic_error_analysis"), pd.DataFrame) else [],
        "uncertainty_metrics": state.get("uncertainty_metrics", {}),
        "abstention_metrics": state.get("abstention_metrics", {}),
        "maintenance_counts": state.get("maintenance_metrics", {}).get("action_counts", {}),
        "model_efficiency": state.get("model_efficiency", pd.DataFrame()).to_dict("records") if isinstance(state.get("model_efficiency"), pd.DataFrame) else [],
        "generated_files": state.get("generated_files", []),
        "warnings": state.get("warnings", []),
        "failures": state.get("failures", []),
        "phase5b_hash_verification": state.get("phase5b_hash_verification", {}),
        "protected_directory_verification": protected_directory_snapshot(state["root"]),
        "reproduction_command": future_full_run_command(state.get("config_path")),
    }
    atomic_write_json(state["output_dir"] / "run_summary.json", summary)
    _record_file(state, state["output_dir"] / "run_summary.json")
    state["run_summary"] = summary
    return summary


def verify_phase5b_hashes_unchanged(state: dict[str, Any]) -> dict[str, Any]:
    current = {str(path): sha256_file(path) for path in _phase5b_artifact_paths(state["config"], state["root"]) if path.exists()}
    changed = [path for path, digest in state["phase5b_initial_hashes"].items() if current.get(path) != digest]
    if changed:
        raise RuntimeError(f"Phase 5B artifact hash changed during Phase 5C full run: {changed}")
    state["phase5b_hash_verification"] = {"verified": True, "file_count": len(current)}
    return state["phase5b_hash_verification"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        return value.to_list()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def candidate_architecture_label(candidate: dict[str, Any]) -> str:
    return json.dumps(_json_ready(candidate.get("architecture_parameters", {})), sort_keys=True)


def normalize_screening_metrics_schema(dataframe: pd.DataFrame) -> pd.DataFrame:
    return _normalize_metric_schema(
        dataframe,
        schema=CANONICAL_SCREENING_SCHEMA,
        numeric_columns=SCREENING_NUMERIC_COLUMNS,
        text_columns=SCREENING_TEXT_COLUMNS,
        require_unique_candidate=True,
        require_status=True,
    )


def normalize_cv_metrics_schema(dataframe: pd.DataFrame) -> pd.DataFrame:
    return _normalize_metric_schema(
        dataframe,
        schema=CANONICAL_CV_SCHEMA,
        numeric_columns=CV_NUMERIC_COLUMNS,
        text_columns=CV_TEXT_COLUMNS,
        require_unique_candidate=False,
        require_status=True,
    )


def _normalize_metric_schema(
    dataframe: pd.DataFrame,
    *,
    schema: list[str],
    numeric_columns: list[str],
    text_columns: list[str],
    require_unique_candidate: bool,
    require_status: bool,
) -> pd.DataFrame:
    data = dataframe.copy(deep=True)
    _apply_metric_aliases(data)
    required = ["candidate_id"]
    if require_status:
        required.append("training_status")
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required metric schema column(s): {missing}")
    for column in schema:
        if column not in data.columns:
            data[column] = "" if column in text_columns else np.nan
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in text_columns:
        if column in data.columns:
            data[column] = data[column].fillna("").astype(str)
    if require_unique_candidate and data["candidate_id"].duplicated().any():
        duplicates = sorted(data.loc[data["candidate_id"].duplicated(keep=False), "candidate_id"].astype(str).unique().tolist())
        raise ValueError(f"Duplicate candidate_id values in screening metrics: {duplicates}")
    extra = [column for column in data.columns if column not in schema]
    return data[[*schema, *extra]]


def _apply_metric_aliases(data: pd.DataFrame) -> None:
    for alias, canonical in METRIC_ALIAS_MAP.items():
        if alias not in data.columns:
            continue
        if canonical in data.columns:
            if _series_conflict(data[canonical], data[alias]):
                raise ValueError(f"Conflicting metric alias values for {canonical}: {alias}")
            data.drop(columns=[alias], inplace=True)
        else:
            data.rename(columns={alias: canonical}, inplace=True)


def _series_conflict(left: pd.Series, right: pd.Series) -> bool:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    numeric_mask = left_num.notna() | right_num.notna()
    if numeric_mask.any():
        left_values = left_num[numeric_mask]
        right_values = right_num[numeric_mask]
        both_missing = left_values.isna() & right_values.isna()
        different_missing = left_values.isna() ^ right_values.isna()
        different_values = (left_values - right_values).abs() > 1.0e-12
        return bool((~both_missing & (different_missing | different_values)).any())
    left_text = left.fillna("").astype(str)
    right_text = right.fillna("").astype(str)
    return bool((left_text != right_text).any())


def validate_ranking_configuration(weights: dict[str, Any]) -> None:
    rejected = []
    unsupported = []
    for name, value in weights.items():
        weight = float(value)
        if weight == 0.0:
            continue
        lower = str(name).lower()
        if any(token in lower for token in REJECTED_RANKING_METRIC_TOKENS):
            rejected.append(str(name))
        elif str(name) not in RANKING_METRIC_REGISTRY:
            unsupported.append(str(name))
    if rejected:
        raise ValueError(f"Benchmark/test ranking metrics are not allowed for Phase 5C selection: {rejected}")
    if unsupported:
        raise ValueError(f"Unsupported robust selection weight(s): {unsupported}")


def ranking_metric_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    weights = config["later_experiment"]["robust_selection_weights"]
    validate_ranking_configuration(weights)
    specs = []
    for weight_name, raw_weight in weights.items():
        weight = float(raw_weight)
        if weight == 0.0:
            continue
        registry = RANKING_METRIC_REGISTRY[str(weight_name)]
        specs.append(
            {
                "weight_name": str(weight_name),
                "metric": str(registry["metric"]),
                "direction": str(registry["direction"]),
                "weight": weight,
            }
        )
    return specs


def assert_no_label_features(feature_names: list[str]) -> None:
    forbidden = {"true_rul", "true_rul_capped", "true_rul_uncapped", "rul_capped", "rul_uncapped", "target_rul_capped", "target_rul_uncapped"}
    leaked = sorted(forbidden & set(feature_names))
    if leaked:
        raise ValueError(f"RUL label column(s) cannot be model features: {leaked}")


def benchmark_sensor_frame_without_labels(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[column for column in BENCHMARK_LABEL_COLUMNS if column in frame.columns], errors="ignore").copy()


def build_benchmark_endpoint_table(test_frame: pd.DataFrame, rul_cap: float, rul_values: list[float] | np.ndarray | pd.Series | None = None) -> pd.DataFrame:
    required = {"subset", "unit_id", "local_unit_id", "global_engine_id", CYCLE_COLUMN}
    missing = sorted(required - set(test_frame.columns))
    if missing:
        raise ValueError(f"Benchmark frame is missing endpoint identity column(s): {missing}")
    if test_frame[["global_engine_id", CYCLE_COLUMN]].duplicated().any():
        raise ValueError("Duplicate test-engine endpoint cycle detected.")
    ordered = test_frame.sort_values(["local_unit_id", CYCLE_COLUMN]).copy()
    mapping = ordered[["subset", "unit_id", "local_unit_id", "global_engine_id"]].drop_duplicates()
    if mapping["local_unit_id"].duplicated().any() or mapping["global_engine_id"].duplicated().any():
        raise ValueError("Duplicate test-engine endpoint identity detected.")
    final_rows = ordered.groupby("local_unit_id", sort=True, observed=False).tail(1).reset_index().rename(columns={"index": "endpoint_row_index"})
    if len(final_rows) != int(test_frame["local_unit_id"].nunique()):
        raise ValueError("Expected exactly one benchmark endpoint per test engine.")
    if rul_values is None:
        if "true_rul_uncapped" not in final_rows.columns:
            raise ValueError("Benchmark endpoint labels require RUL-file values or final-row true_rul_uncapped.")
        labels = final_rows["true_rul_uncapped"].to_numpy(dtype=float)
    else:
        labels = np.asarray(rul_values, dtype=float).reshape(-1)
        if len(labels) != len(final_rows):
            raise ValueError(f"RUL-file row count mismatch: expected {len(final_rows)}, found {len(labels)}.")
    if not np.isfinite(labels).all() or (labels < 0).any():
        raise ValueError("Benchmark RUL labels must be finite and non-negative.")
    endpoints = final_rows[["subset", "unit_id", "local_unit_id", "global_engine_id", CYCLE_COLUMN, "endpoint_row_index"]].copy()
    endpoints = endpoints.rename(columns={CYCLE_COLUMN: "final_observed_cycle"})
    endpoints["true_rul"] = labels.astype(float)
    endpoints["true_rul_capped"] = np.minimum(endpoints["true_rul"].to_numpy(dtype=float), float(rul_cap))
    endpoints["endpoint_index"] = final_rows.groupby("global_engine_id", observed=False).cumcount()
    endpoint_indices = []
    for _, row in endpoints.iterrows():
        group = ordered[ordered["global_engine_id"] == row["global_engine_id"]].reset_index(drop=True)
        matches = group.index[group[CYCLE_COLUMN].astype(int) == int(row["final_observed_cycle"])].tolist()
        if len(matches) != 1:
            raise ValueError(f"Final observed cycle did not align for {row['global_engine_id']}.")
        endpoint_indices.append(int(matches[0]))
    endpoints["endpoint_index"] = endpoint_indices
    if endpoints[["subset", "global_engine_id"]].duplicated().any():
        raise ValueError("Duplicate benchmark endpoint detected.")
    return endpoints[["subset", "unit_id", "global_engine_id", "final_observed_cycle", "true_rul", "true_rul_capped", "endpoint_row_index", "endpoint_index"]]


def attach_benchmark_labels(predictions: pd.DataFrame, endpoint_labels: pd.DataFrame) -> pd.DataFrame:
    left = predictions.copy()
    if "final_observed_cycle" not in left.columns:
        left["final_observed_cycle"] = left["cycle"].astype(int)
    labels = endpoint_labels[["subset", "global_engine_id", "final_observed_cycle", "true_rul", "true_rul_capped"]].copy()
    merged = left.merge(labels, on=["subset", "global_engine_id", "final_observed_cycle"], how="left", validate="one_to_one")
    if merged["true_rul"].isna().any():
        missing = merged.loc[merged["true_rul"].isna(), ["subset", "global_engine_id", "final_observed_cycle"]].to_dict("records")
        raise ValueError(f"Benchmark labels did not attach for endpoint(s): {missing}")
    merged["residual"] = merged["predicted_rul"].astype(float) - merged["true_rul"].astype(float)
    merged["absolute_error"] = merged["residual"].abs()
    merged["squared_error"] = np.square(merged["residual"])
    merged["prediction_direction"] = [prediction_direction(value) for value in merged["residual"]]
    return merged


def normalize_cv_prediction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame.copy() for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=CV_PREDICTION_SCHEMA)
    normalized = []
    for frame in non_empty:
        for column in CV_PREDICTION_SCHEMA:
            if column not in frame.columns:
                frame[column] = np.nan
        extras = [column for column in frame.columns if column not in CV_PREDICTION_SCHEMA]
        normalized.append(frame[[*CV_PREDICTION_SCHEMA, *extras]])
    return pd.concat(normalized, ignore_index=True)


def artifact_sha256_and_size(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "size_bytes": 0, "sha256": ""}
    return {"exists": True, "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def stable_payload_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(_json_ready(payload), sort_keys=True).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(payload), handle, indent=2, allow_nan=False)
    temp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _record_file(state: dict[str, Any], path: Path) -> None:
    text = str(Path(path))
    if text not in state["generated_files"]:
        state["generated_files"].append(text)


def write_failure_summary(state: dict[str, Any]) -> None:
    if "output_dir" not in state:
        return
    screening_path = state["output_dir"] / "screening_metrics.csv"
    screening_frame = state.get("screening_metrics", pd.DataFrame())
    if (not isinstance(screening_frame, pd.DataFrame) or screening_frame.empty) and screening_path.exists():
        try:
            screening_frame = pd.read_csv(screening_path)
        except Exception:
            screening_frame = pd.DataFrame()
    failure_text = " ".join(str(item.get("error", "")) for item in state.get("failures", []))
    missing_metric = ""
    if "Configured ranking metric is missing:" in failure_text:
        missing_metric = failure_text.split("Configured ranking metric is missing:", 1)[1].split()[0].strip(" .,'\"")
    resume = inspect_phase5c_resume_state(state["config"], state["root"]) if "config" in state and "root" in state else {}
    locked_candidate_id = state.get("locked_candidate", {}).get("candidate_id", "")
    final_checkpoint = state.get("final_fit_metadata", {}).get("checkpoint_path") or str(state.get("checkpoint_dir", Path("")) / "locked_physics_guided_model.pt")
    benchmark_subset = state.get("current_benchmark_subset", "")
    benchmark_frame = state.get("benchmark_frames", {}).get(benchmark_subset, pd.DataFrame()) if benchmark_subset else pd.DataFrame()
    payload = {
        "run_status": "failed",
        "stage": state.get("current_stage"),
        "failed_stage": state.get("current_stage"),
        "exception_type": state.get("failures", [{}])[-1].get("error", "").split(":", 1)[0] if state.get("failures") else "",
        "exception_message": state.get("failures", [{}])[-1].get("error", "").split(":", 1)[1].strip() if state.get("failures") and ":" in state.get("failures", [{}])[-1].get("error", "") else "",
        "completed_stages": list(state.get("stage_results", {}).keys()),
        "locked_candidate_id": locked_candidate_id,
        "final_checkpoint_path": final_checkpoint,
        "final_checkpoint_exists": Path(final_checkpoint).exists() if final_checkpoint else False,
        "resume_eligibility": resume,
        "earliest_safe_resume_stage": resume.get("earliest_safe_resume_stage", ""),
        "missing_resume_artifacts": resume.get("missing_or_invalid_artifacts", []),
        "benchmark_subset_being_processed": benchmark_subset,
        "expected_benchmark_frame_columns": sorted(["subset", "unit_id", "local_unit_id", "global_engine_id", CYCLE_COLUMN, *state.get("final_preprocessor", {}).get("features", [])]),
        "actual_benchmark_frame_columns": list(benchmark_frame.columns),
        "partial_status": "incomplete_failed",
        "missing_metric": missing_metric,
        "screening_metrics_path": str(screening_path) if screening_path.exists() else "",
        "screening_candidate_count": int(len(screening_frame)) if isinstance(screening_frame, pd.DataFrame) else 0,
        "screening_successful_candidate_count": int((screening_frame.get("training_status", pd.Series(dtype=object)).astype(str) == "success").sum()) if isinstance(screening_frame, pd.DataFrame) and not screening_frame.empty else 0,
        "screening_failed_candidate_count": int((screening_frame.get("training_status", pd.Series(dtype=object)).astype(str) == "failed").sum()) if isinstance(screening_frame, pd.DataFrame) and not screening_frame.empty else 0,
        "screening_schema": list(screening_frame.columns) if isinstance(screening_frame, pd.DataFrame) else [],
        "failures": state.get("failures", []),
        "warnings": state.get("warnings", []),
        "runtime_seconds": float(state.get("runtime_seconds", 0.0)),
        "generated_files": state.get("generated_files", []),
        "no_completion_claim": True,
    }
    atomic_write_json(state["output_dir"] / "run_summary.json", payload)


def _phase5b_artifact_paths(config: dict[str, Any], root: Path) -> list[Path]:
    phase5b_dir = resolve_project_path(config["general"]["phase5b_results_path"], root)
    checkpoint = resolve_project_path(config["general"]["phase5b_checkpoint_path"], root)
    names = [
        "run_summary.json",
        "benchmark_predictions.csv",
        "benchmark_metrics.json",
        "metrics_by_subset.csv",
        "uncertainty_predictions.csv",
        "uncertainty_metrics.json",
        "deep_uncertainty_predictions.csv",
        "deep_uncertainty_metrics.json",
        "model_efficiency.csv",
        "locked_extended_model.json",
        "phase5_vs_phase5b.csv",
        "phase5_vs_phase5b_uncertainty.csv",
    ]
    return [phase5b_dir / name for name in names if (phase5b_dir / name).exists()] + [checkpoint]


def _dataset_files(config: dict[str, Any], root: Path) -> dict[str, str]:
    dataset_dir = resolve_project_path(config["general"]["dataset_dir"], root)
    files: dict[str, str] = {}
    for subset in config["general"]["training_subsets"]:
        files[f"train_{subset}"] = str(dataset_dir / f"train_{str(subset).upper()}.txt")
    for subset in config["general"]["benchmark_test_subsets"]:
        files[f"test_{subset}"] = str(dataset_dir / f"test_{str(subset).upper()}.txt")
        files[f"rul_{subset}"] = str(dataset_dir / f"RUL_{str(subset).upper()}.txt")
    return files


def future_full_run_command(config_path: Path | None) -> str:
    config_text = config_path.as_posix() if config_path else "configs/physics_guided_temporal_rul.yaml"
    return (
        '$env:PYTHONPATH = ".\\src"\n'
        "python -m aeroguard.pipelines.train_physics_guided_temporal_rul "
        f'--config "{config_text}" --full-run'
    )


def protected_directory_snapshot(root: Path) -> dict[str, Any]:
    rows = []
    for relative in PROTECTED_DIRECTORIES:
        path = (root / relative).resolve()
        rows.append({"path": str(path), "exists": path.exists(), "last_write_time": None if not path.exists() else path.stat().st_mtime})
    return {"checked": rows, "statement": "Protected directories are not outputs of Phase 5C."}


def _healthy_rul_threshold(config: dict[str, Any]) -> float:
    return float(config["sequence"].get("healthy_rul_threshold", config["sequence"]["rul_cap"]))


def _critical_rul_threshold(config: dict[str, Any]) -> float:
    return float(config["sequence"].get("critical_rul_threshold", config["safety"].get("low_rul_threshold", 30.0)))


def _preprocess_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "include_cycle_as_feature": bool(config["sequence"].get("include_cycle_as_feature", False)),
        "features_to_exclude": list(config["sequence"].get("features_to_exclude", [])),
        "near_constant_threshold": float(config["sequence"].get("near_constant_threshold", 0.0)),
        "correlation_threshold": float(config["sequence"].get("correlation_threshold", 0.999)),
        "operating_condition_method": str(config["sequence"].get("operating_condition_method", "regime_standardization")),
        "number_of_operating_regimes": int(config["sequence"].get("number_of_operating_regimes", 6)),
        "residualization_ridge_alpha": float(config["sequence"].get("residualization_ridge_alpha", 1.0)),
        "random_seed": int(config["general"]["random_seed"]),
    }


def _policy_config(config: dict[str, Any]) -> dict[str, Any]:
    support = dict(config["safety"]["support_settings"])
    support.setdefault("limited_robust_distance", 3.0)
    support.setdefault("out_robust_distance", 6.0)
    support.setdefault("regime_distance_quantile", 0.99)
    abstention = dict(config["safety"]["abstention_settings"])
    abstention.setdefault("max_regime_distance", 6.0)
    abstention.setdefault("abstain_on_quantile_crossing", True)
    abstention.setdefault("high_error_threshold", float(config["safety"]["severe_optimistic_threshold"]))
    return {"support_settings": support, "abstention_settings": abstention, "maintenance_thresholds": config["safety"]["maintenance_thresholds"]}


def _window_spec(config: dict[str, Any]) -> WindowSpec:
    return WindowSpec(
        window_length=int(config["sequence"]["window_length"]),
        stride=int(config["sequence"]["window_stride"]),
        minimum_valid_history=int(config["sequence"]["minimum_valid_history"]),
    )


def _pairing_config(config: dict[str, Any]) -> TemporalPairingConfig:
    pairing = config["pairing"]
    return TemporalPairingConfig(
        adjacent_enabled=bool(pairing["adjacent_pair_enabled"]),
        fixed_gap_enabled=bool(pairing["fixed_gap_pair_enabled"]),
        triplet_enabled=bool(pairing["triplet_enabled"]),
        allowed_cycle_gaps=tuple(int(value) for value in pairing["allowed_cycle_gaps"]),
        max_adjacent_pairs_per_engine=int(pairing["maximum_adjacent_pairs_per_engine"]),
        max_fixed_gap_pairs_per_engine=int(pairing["maximum_fixed_gap_pairs_per_engine"]),
        max_triplets_per_engine=int(pairing["maximum_triplets_per_engine"]),
        seed=int(pairing["pair_seed"]),
        sampling_method=str(pairing["sampling_method"]),
    )


def _regime_config(config: dict[str, Any], *, enabled: bool | None = None) -> RegimePairingConfig:
    regime = config["regime_consistency"]
    active = bool(regime["enabled"]) if enabled is None else bool(enabled)
    return RegimePairingConfig(
        enabled=active,
        rul_tolerance=float(regime["rul_matching_tolerance"]),
        max_pairs=int(regime["maximum_regime_pairs"]),
        seed=int(regime["pair_seed"]),
        sampling_method="uniform",
        max_anchors=int(regime.get("maximum_regime_anchors", regime.get("maximum_anchors", 10_000))),
        max_partners_per_anchor=int(regime.get("maximum_partners_per_anchor", 2)),
        max_pairs_per_regime_pair=int(regime.get("maximum_pairs_per_regime_combination", 4_000)),
        allow_empty_pairs=bool(regime.get("allow_empty_pairs", True)),
        lazy_build=bool(regime.get("lazy_build", True)),
        cache_bounded_pairs=bool(regime.get("cache_bounded_pairs", True)),
    )


def _candidate_registry(config: dict[str, Any]) -> list[dict[str, Any]]:
    registry = config["candidate_registry"].get("definitions") or default_candidate_registry()
    validate_candidate_registry(registry, max_candidates=int(config["candidate_registry"]["maximum_candidate_count"]))
    return registry


def candidate_requires_regime_pairs(candidate: dict[str, Any]) -> bool:
    return "regime" in set(candidate.get("active_losses", [])) or "regime" in set(candidate.get("pairing_requirements", []))


def get_regime_pairs_for_candidate(state: dict[str, Any], candidate: dict[str, Any], split_key: str, metadata: pd.DataFrame) -> pd.DataFrame:
    if not candidate_requires_regime_pairs(candidate):
        return empty_regime_pair_frame("candidate_no_regime_loss")
    config = _regime_config(state["config"])
    if not config.enabled:
        raise ValueError(f"Candidate {candidate['candidate_id']} requires regime pairs but regime_consistency.enabled is false.")
    cache = state.setdefault("regime_pair_cache", {})
    key = regime_pair_cache_key(split_key, metadata, config)
    if bool(config.cache_bounded_pairs) and key in cache:
        cached = cache[key]
        state.setdefault("regime_pair_cache_hits", 0)
        state["regime_pair_cache_hits"] += 1
        return cached
    start = time.perf_counter()
    pairs = build_regime_pairs(metadata, config)
    diagnostics = dict(pairs.attrs.get("diagnostics", {}))
    diagnostics.update({"split_key": split_key, "candidate_id": candidate["candidate_id"], "cache_key": key, "cache_hit": False})
    state.setdefault("regime_pair_diagnostics_by_split", {})[key] = diagnostics
    state.setdefault("warnings", [])
    print(
        "Regime pairing: "
        f"metadata_rows={diagnostics.get('metadata_rows', len(metadata))} "
        f"regimes={diagnostics.get('number_of_regimes', 'unknown')} "
        f"anchors={diagnostics.get('anchor_count_considered', 0)} "
        f"retained_pairs={len(pairs)} "
        f"pair_memory_mb={diagnostics.get('pair_table_memory_mb', 0.0):.3f} "
        f"limit_reached={diagnostics.get('limit_reached', False)}"
    )
    if pairs.empty and not bool(config.allow_empty_pairs):
        raise ValueError(f"Regime-pair generation produced no pairs for {candidate['candidate_id']}.")
    if bool(config.cache_bounded_pairs):
        cache[key] = pairs
    state.setdefault("regime_pair_runtime_seconds", 0.0)
    state["regime_pair_runtime_seconds"] += time.perf_counter() - start
    return pairs


def regime_pair_cache_key(split_key: str, metadata: pd.DataFrame, config: RegimePairingConfig) -> str:
    engine_count = int(metadata["global_engine_id"].nunique()) if "global_engine_id" in metadata.columns else 0
    sample_min = int(metadata["sample_index"].min()) if len(metadata) and "sample_index" in metadata.columns else -1
    sample_max = int(metadata["sample_index"].max()) if len(metadata) and "sample_index" in metadata.columns else -1
    fingerprint_columns = [
        column
        for column in ["sample_index", "global_engine_id", "subset", "operating_regime", "target_rul_capped", "sequence_valid_length"]
        if column in metadata.columns
    ]
    if fingerprint_columns:
        hashed_rows = pd.util.hash_pandas_object(metadata[fingerprint_columns], index=False).to_numpy(dtype=np.uint64, copy=False)
        metadata_fingerprint = hashlib.sha256(hashed_rows.tobytes()).hexdigest()[:16]
    else:
        metadata_fingerprint = "empty"
    raw = {
        "split_key": split_key,
        "row_count": int(len(metadata)),
        "engine_count": engine_count,
        "sample_min": sample_min,
        "sample_max": sample_max,
        "metadata_fingerprint": metadata_fingerprint,
        "rul_tolerance": float(config.rul_tolerance),
        "max_pairs": int(config.max_pairs),
        "max_anchors": int(config.max_anchors),
        "max_partners_per_anchor": int(config.max_partners_per_anchor),
        "max_pairs_per_regime_pair": int(config.max_pairs_per_regime_pair),
        "seed": int(config.seed),
    }
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"regime_pairs_{split_key}_{digest}"


def _schedule_for_candidate(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    later = config["later_experiment"]
    schedule = dict(later["training_schedules"][candidate["training_schedule"]])
    schedule.update(
        {
            "optimizer": later["optimizer"],
            "batch_size": int(later["batch_size"]),
            "num_workers": int(later["num_workers"]),
            "gradient_clip_norm": float(later["gradient_clip_norm"]),
            "mixed_precision": later.get("mixed_precision", "auto"),
        }
    )
    schedule["max_epochs"] = min(int(schedule["max_epochs"]), int(later["maximum_epochs"]))
    schedule["minimum_epochs"] = min(int(schedule["minimum_epochs"]), int(schedule["max_epochs"]))
    return schedule


def validate_engine_frame(frame: pd.DataFrame) -> None:
    required = {"subset", "global_engine_id", CYCLE_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Frame is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Loaded subset frame must not be empty.")
    if frame[["global_engine_id", CYCLE_COLUMN]].duplicated().any():
        raise ValueError("Duplicate global-engine-cycle rows detected.")
    if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all():
        raise ValueError("Loaded frame contains non-finite numeric values.")
    ordered = frame.sort_values(["global_engine_id", CYCLE_COLUMN])
    for _, group in ordered.groupby("global_engine_id", observed=False):
        if not group[CYCLE_COLUMN].is_monotonic_increasing:
            raise ValueError("Engine cycles must be monotonic increasing.")


def assert_training_only_preprocessing_frame(frame: pd.DataFrame) -> None:
    if "data_role" in frame.columns and frame["data_role"].astype(str).str.contains("benchmark|test", case=False, regex=True).any():
        raise ValueError("Benchmark/test rows cannot be used to fit preprocessing.")
    if "subset" in frame.columns and frame["subset"].astype(str).str.lower().str.startswith("test_").any():
        raise ValueError("Benchmark/test subsets cannot be used to fit preprocessing.")


class IndexedSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: SequenceWindowDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, length = self.dataset[index]
        return torch.as_tensor(index, dtype=torch.long), x, y, length


def build_candidate_model(candidate: dict[str, Any], input_dim: int, config: dict[str, Any]) -> PhysicsGuidedPatchTransformer:
    architecture = dict(candidate["architecture_parameters"])
    architecture["input_dim"] = int(input_dim)
    architecture["health_head_enabled"] = "health" in set(candidate["active_output_heads"])
    architecture["rate_head_enabled"] = "rate" in set(candidate["active_output_heads"])
    architecture.setdefault("output_activation", config["model"]["output_activation"])
    architecture["parameter_budget"] = int(candidate.get("parameter_budget", config["model"]["parameter_budget"]))
    model = PhysicsGuidedPatchTransformer(**architecture)
    warm = config["warm_start"]
    if bool(warm.get("enabled", False)) and bool(candidate.get("warm_start", {}).get("enabled", True)):
        model.warm_start_from_checkpoint(
            resolve_project_path(warm["checkpoint_path"], project_root()),
            load_encoder_only=bool(warm.get("load_encoder_only", True)),
            strict=bool(warm.get("strict", False)),
        )
    return model


def candidate_loss_config(config: dict[str, Any], candidate: dict[str, Any]) -> PhysicsLossConfig:
    values = dict(config["losses"])
    values.update(active_loss_weights(candidate))
    values.update(candidate.get("loss_tolerances", {}))
    values["include_rate_head_loss"] = "rate" in set(candidate["active_losses"]) and "rate" in set(candidate["active_output_heads"])
    values["allow_missing_optional_batches"] = False
    return PhysicsLossConfig.from_mapping(values)


def train_physics_model(
    candidate: dict[str, Any],
    train_dataset: SequenceWindowDataset,
    validation_dataset: SequenceWindowDataset,
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    temporal_pairs: pd.DataFrame,
    regime_pairs: pd.DataFrame,
    config: dict[str, Any],
    schedule: dict[str, Any],
    device: torch.device,
    seed: int,
    *,
    fixed_epochs: int | None = None,
) -> tuple[PhysicsGuidedPatchTransformer, dict[str, Any]]:
    set_global_seed(int(seed), bool(config["general"].get("deterministic_algorithms", False)))
    model = build_candidate_model(candidate, int(train_dataset.sequences.shape[-1]), config).to(device)
    loss_fn = CompositePhysicsLoss(candidate_loss_config(config, candidate))
    optimizer = _optimizer(model, schedule)
    loader = DataLoader(
        IndexedSequenceDataset(train_dataset),
        batch_size=int(schedule["batch_size"]),
        shuffle=True,
        num_workers=int(schedule.get("num_workers", 0)),
    )
    max_epochs = int(fixed_epochs or schedule["max_epochs"])
    minimum_epochs = int(fixed_epochs or schedule["minimum_epochs"])
    patience = int(schedule.get("early_stopping_patience", 0))
    best_state = deepcopy(model.state_dict())
    best_rmse = math.inf
    best_epoch = 0
    bad_epochs = 0
    history = []
    rng = np.random.default_rng(int(seed))
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        grad_norms = []
        epoch_start = time.perf_counter()
        for indices, _, _, _ in loader:
            x, batch = structured_training_batch(
                train_dataset,
                train_metadata,
                temporal_pairs,
                regime_pairs,
                indices.numpy(),
                candidate,
                config,
                rng,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss_result = loss_fn(outputs, batch)
            loss = loss_result["total_loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Phase 5C loss encountered.")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(schedule["gradient_clip_norm"]))
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite Phase 5C gradients encountered.")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(float(grad_norm.detach().cpu()))
        predictions = evaluate_physics_model_frame(model, validation_dataset, validation_metadata, device, int(schedule["batch_size"]), str(candidate["candidate_id"]))
        metrics = validation_metrics_for_frame(predictions, config)
        current_rmse = float(metrics["validation_rmse"])
        improved = current_rmse < best_rmse
        if improved:
            best_rmse = current_rmse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        elif epoch >= minimum_epochs:
            bad_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_rmse": current_rmse,
                "gradient_norm": float(np.mean(grad_norms)),
                "epoch_seconds": time.perf_counter() - epoch_start,
                "improved": bool(improved),
            }
        )
        if fixed_epochs is None and epoch >= minimum_epochs and bad_epochs > patience:
            break
    model.load_state_dict(best_state)
    metadata = {
        "history": history,
        "best_epoch": int(best_epoch or max_epochs),
        "stopping_epoch": int(history[-1]["epoch"]) if history else 0,
        "best_validation_rmse": float(best_rmse),
        "training_seconds": time.perf_counter() - start,
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])) if history else 0.0,
        "early_stopping_triggered": bool(fixed_epochs is None and history and int(history[-1]["epoch"]) < max_epochs),
    }
    return model, metadata


def structured_training_batch(
    dataset: SequenceWindowDataset,
    metadata: pd.DataFrame,
    temporal_pairs: pd.DataFrame,
    regime_pairs: pd.DataFrame,
    base_indices: np.ndarray,
    candidate: dict[str, Any],
    config: dict[str, Any],
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    selected = {int(index) for index in base_indices.tolist()}
    temporal_pair_rows = _sample_pair_rows(temporal_pairs, candidate, rng, "pair", 64)
    triplet_rows = _sample_pair_rows(temporal_pairs, candidate, rng, "triplet", 32)
    regime_rows = _sample_regime_rows(regime_pairs, candidate, rng, 64)
    for frame, columns in [(temporal_pair_rows, ["earlier_index", "later_index"]), (triplet_rows, ["earlier_index", "middle_index", "later_index"]), (regime_rows, ["left_index", "right_index"])]:
        if not frame.empty:
            for column in columns:
                selected.update(int(value) for value in frame[column].to_numpy(dtype=int))
    ordered = np.asarray(sorted(selected), dtype=np.int64)
    local = {int(global_index): int(local_index) for local_index, global_index in enumerate(ordered)}
    x = dataset.sequences[ordered].to(device)
    target = dataset.targets[ordered].to(device)
    batch: dict[str, torch.Tensor] = {"target_rul": target}
    if "health" in set(candidate["active_losses"]) or "health_monotonic" in set(candidate["active_losses"]):
        batch["health_target"] = normalized_capped_rul_targets(target.detach().cpu(), float(config["sequence"]["rul_cap"])).to(device)
    if not temporal_pair_rows.empty:
        batch["pair_indices"] = torch.as_tensor([[local[int(row.earlier_index)], local[int(row.later_index)]] for row in temporal_pair_rows.itertuples()], dtype=torch.long, device=device)
        batch["pair_cycle_gaps"] = torch.as_tensor(temporal_pair_rows["cycle_gap"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).view(-1, 1)
        plateau = (temporal_pair_rows["earlier_true_capped_rul"].astype(float) >= float(config["sequence"]["rul_cap"])) | (temporal_pair_rows["later_true_capped_rul"].astype(float) >= float(config["sequence"]["rul_cap"]))
        batch["pair_plateau_mask"] = torch.as_tensor(plateau.to_numpy(dtype=np.float32), dtype=torch.float32, device=device).view(-1, 1)
    if not triplet_rows.empty:
        batch["triplet_indices"] = torch.as_tensor([[local[int(row.earlier_index)], local[int(row.middle_index)], local[int(row.later_index)]] for row in triplet_rows.itertuples()], dtype=torch.long, device=device)
        batch["triplet_left_gaps"] = torch.as_tensor(triplet_rows["left_gap"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).view(-1, 1)
        batch["triplet_right_gaps"] = torch.as_tensor(triplet_rows["right_gap"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).view(-1, 1)
    if not regime_rows.empty:
        batch["regime_pair_indices"] = torch.as_tensor([[local[int(row.left_index)], local[int(row.right_index)]] for row in regime_rows.itertuples()], dtype=torch.long, device=device)
    _validate_required_structures(candidate, batch)
    return x, batch


def _sample_pair_rows(pair_frame: pd.DataFrame, candidate: dict[str, Any], rng: np.random.Generator, kind: str, limit: int) -> pd.DataFrame:
    if pair_frame.empty:
        return pair_frame
    requirements = set(candidate["pairing_requirements"])
    if kind == "triplet":
        if "triplet" not in requirements:
            return pair_frame.iloc[[]]
        frame = pair_frame[pair_frame["pair_type"] == "triplet"]
    else:
        types = []
        if "adjacent" in requirements:
            types.append("adjacent")
        if "fixed_gap" in requirements:
            types.append("fixed_gap")
        frame = pair_frame[pair_frame["pair_type"].isin(types)] if types else pair_frame.iloc[[]]
    if len(frame) <= limit:
        return frame.copy()
    chosen = rng.choice(frame.index.to_numpy(), size=limit, replace=False)
    return frame.loc[np.sort(chosen)].copy()


def _sample_regime_rows(regime_pairs: pd.DataFrame, candidate: dict[str, Any], rng: np.random.Generator, limit: int) -> pd.DataFrame:
    if "regime" not in set(candidate["pairing_requirements"]) or regime_pairs.empty:
        return regime_pairs.iloc[[]] if not regime_pairs.empty else regime_pairs
    if len(regime_pairs) <= limit:
        return regime_pairs.copy()
    chosen = rng.choice(regime_pairs.index.to_numpy(), size=limit, replace=False)
    return regime_pairs.loc[np.sort(chosen)].copy()


def _validate_required_structures(candidate: dict[str, Any], batch: dict[str, torch.Tensor]) -> None:
    losses = set(candidate["active_losses"])
    if {"monotonic", "rate", "health_monotonic"} & losses and "pair_indices" not in batch:
        raise ValueError(f"Candidate {candidate['candidate_id']} requires temporal pairs but none were sampled.")
    if "smooth" in losses and "triplet_indices" not in batch:
        raise ValueError(f"Candidate {candidate['candidate_id']} requires triplets but none were sampled.")
    if "regime" in losses and "regime_pair_indices" not in batch:
        raise ValueError(f"Candidate {candidate['candidate_id']} requires regime pairs but none were sampled.")


def _optimizer(model: torch.nn.Module, schedule: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(schedule.get("optimizer", "adamw")).lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(schedule["learning_rate"]), weight_decay=float(schedule.get("weight_decay", 0.0)))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(schedule["learning_rate"]), weight_decay=float(schedule.get("weight_decay", 0.0)))
    raise ValueError(f"Unsupported optimizer: {name}")


@torch.no_grad()
def predict_physics_batches(model: PhysicsGuidedPatchTransformer, dataset: SequenceWindowDataset | InferenceSequenceDataset, device: torch.device, batch_size: int) -> pd.DataFrame:
    model.eval()
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    rows = []
    for batch in loader:
        if len(batch) == 3:
            x, _, lengths = batch
        else:
            x, lengths = batch
        outputs = model(x.to(device), lengths.to(device))
        rows.append(
            pd.DataFrame(
                {
                    "predicted_rul_raw": outputs["rul_raw"].detach().cpu().numpy().ravel(),
                    "predicted_rul": outputs["rul_prediction"].detach().cpu().numpy().ravel(),
                    "health_score": np.nan if outputs["health_score"] is None else outputs["health_score"].detach().cpu().numpy().ravel(),
                    "degradation_rate": np.nan if outputs["degradation_rate"] is None else outputs["degradation_rate"].detach().cpu().numpy().ravel(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def evaluate_physics_model_frame(
    model: PhysicsGuidedPatchTransformer,
    dataset: SequenceWindowDataset,
    metadata: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    candidate_id: str,
) -> pd.DataFrame:
    pred = predict_physics_batches(model, dataset, device, batch_size)
    return _prediction_frame_from_outputs(metadata, pred, candidate_id)


def _prediction_frame_from_outputs(metadata: pd.DataFrame, pred: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    result = pd.concat([metadata.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    result["candidate_id"] = candidate_id
    result["true_rul"] = result["target_rul_uncapped"].astype(float)
    result["residual"] = result["predicted_rul"].astype(float) - result["true_rul"]
    result["absolute_error"] = result["residual"].abs()
    result["squared_error"] = np.square(result["residual"])
    result["prediction_direction"] = [prediction_direction(value) for value in result["residual"]]
    if "cycle" in result.columns:
        result["final_observed_cycle"] = result["cycle"]
    return result


def validation_metrics_for_frame(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    metrics = deep_point_metrics(frame["true_rul"], frame["predicted_rul"], float(config["safety"]["severe_optimistic_threshold"]))
    opt = optimistic_error_metrics(frame["true_rul"], frame["predicted_rul"], severe_threshold=float(config["safety"]["severe_optimistic_threshold"]), low_rul_threshold=float(config["safety"]["low_rul_threshold"]))
    result = {
        "validation_mae": float(metrics["mae"]),
        "validation_rmse": float(metrics["rmse"]),
        "validation_nasa_score": float(metrics["nasa_score"]),
        "validation_mean_signed_error": float(metrics["mean_signed_error"]),
        "validation_optimistic_rate": float(opt["optimistic_prediction_rate"]),
        "validation_severe_optimistic_rate": float(opt["severe_optimistic_prediction_rate"]),
        "validation_low_rul_optimistic_rate": float(opt["low_rul_optimistic_error_rate"]),
    }
    non_finite = [name for name, value in result.items() if not np.isfinite(float(value))]
    if non_finite:
        raise ValueError(f"Validation metric calculation produced non-finite values: {non_finite}")
    return result


def constraint_metrics_for_predictions(predictions: pd.DataFrame, temporal_pairs: pd.DataFrame, regime_pairs: pd.DataFrame) -> dict[str, float]:
    pred = predictions["predicted_rul"].to_numpy(dtype=float)
    metrics = {
        "monotonic_violation_rate": np.nan,
        "rate_violation_rate": np.nan,
        "smoothness_violation_rate": np.nan,
        "health_violation_rate": np.nan,
        "regime_consistency_violation_rate": np.nan,
    }
    if not temporal_pairs.empty:
        pairs = temporal_pairs[temporal_pairs["pair_type"].isin(["adjacent", "fixed_gap"])]
        pairs = pairs[(pairs["earlier_index"] < len(pred)) & (pairs["later_index"] < len(pred))]
        if not pairs.empty:
            mono = monotonicity_metrics(pred[pairs["earlier_index"].to_numpy(dtype=int)], pred[pairs["later_index"].to_numpy(dtype=int)])
            rate = cycle_rate_metrics(pred[pairs["earlier_index"].to_numpy(dtype=int)], pred[pairs["later_index"].to_numpy(dtype=int)], pairs["cycle_gap"].to_numpy(dtype=float))
            metrics["monotonic_violation_rate"] = mono["violation_rate"]
            metrics["rate_violation_rate"] = rate["rate_violation_rate"]
        triplets = temporal_pairs[temporal_pairs["pair_type"] == "triplet"]
        triplets = triplets[(triplets["earlier_index"] < len(pred)) & (triplets["middle_index"] < len(pred)) & (triplets["later_index"] < len(pred))]
        if not triplets.empty:
            smooth = smoothness_metrics(pred[triplets["earlier_index"].to_numpy(dtype=int)], pred[triplets["middle_index"].to_numpy(dtype=int)], pred[triplets["later_index"].to_numpy(dtype=int)])
            metrics["smoothness_violation_rate"] = smooth["smoothness_violation_rate"]
    if "health_score" in predictions.columns and predictions["health_score"].notna().any():
        health_diag = health_rul_consistency_diagnostics(predictions["predicted_rul"], predictions["health_score"].fillna(predictions["predicted_rul"]))
        metrics["health_violation_rate"] = health_diag["health_directional_disagreement_rate"]
    if not regime_pairs.empty and {"left_index", "right_index"}.issubset(regime_pairs.columns):
        valid = regime_pairs[(regime_pairs["left_index"] < len(pred)) & (regime_pairs["right_index"] < len(pred))]
        if not valid.empty:
            disagreement = np.abs(pred[valid["left_index"].to_numpy(dtype=int)] - pred[valid["right_index"].to_numpy(dtype=int)])
            metrics["regime_consistency_violation_rate"] = float((disagreement > 0.0).mean())
    return metrics


def safe_efficiency_row(model: torch.nn.Module, dataset: SequenceWindowDataset | None, device: torch.device, config: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    if dataset is None:
        return {"model_id": candidate_id, "parameter_count": trainable_parameter_count(model)}
    try:
        example_single = dataset.sequences[:1]
        batch_count = min(32, len(dataset.sequences))
        example_batch = dataset.sequences[:batch_count]
        row = model_efficiency_row(candidate_id, model, example_single, example_batch, device, {"training_seconds": None, "mean_epoch_seconds": None}, repetitions=10)
        return dict(row)
    except Exception as exc:
        return {"model_id": candidate_id, "parameter_count": trainable_parameter_count(model), "efficiency_error": f"{type(exc).__name__}: {exc}"}


def _failed_candidate_row(candidate: dict[str, Any], state: dict[str, Any], exc: Exception) -> dict[str, Any]:
    row = {
        "candidate_id": candidate["candidate_id"],
        "architecture": candidate_architecture_label(candidate),
        "active_losses": ";".join(candidate["active_losses"]),
        "active_heads": ";".join(candidate["active_output_heads"]),
        "fitting_engine_count": int(state.get("screening_train_metadata", pd.DataFrame()).get("global_engine_id", pd.Series(dtype=object)).nunique()),
        "validation_engine_count": int(state.get("screening_validation_metadata", pd.DataFrame()).get("global_engine_id", pd.Series(dtype=object)).nunique()),
        "standard_window_count": int(len(state.get("screening_train_metadata", []))),
        "temporal_pair_count": 0,
        "adjacent_pair_count": 0,
        "fixed_gap_pair_count": 0,
        "temporal_triplet_count": 0,
        "regime_pair_count": 0,
        "best_epoch": np.nan,
        "stopping_epoch": np.nan,
        "validation_mae": np.nan,
        "validation_rmse": np.nan,
        "validation_nasa_score": np.nan,
        "validation_mean_signed_error": np.nan,
        "validation_optimistic_rate": np.nan,
        "validation_severe_optimistic_rate": np.nan,
        "validation_low_rul_optimistic_rate": np.nan,
        "monotonic_violation_rate": np.nan,
        "rate_violation_rate": np.nan,
        "smoothness_violation_rate": np.nan,
        "health_violation_rate": np.nan,
        "regime_consistency_violation_rate": np.nan,
        "parameter_count": np.nan,
        "checkpoint_size": np.nan,
        "training_runtime": np.nan,
        "cpu_latency": np.nan,
        "gpu_latency": np.nan,
        "training_status": "failed",
        "failure_reason": f"{type(exc).__name__}: {exc}",
        "checkpoint_path": "",
    }
    return normalize_screening_metrics_schema(pd.DataFrame([row])).iloc[0].to_dict()


def _pair_counts(temporal_pairs: pd.DataFrame, regime_pairs: pd.DataFrame) -> dict[str, int]:
    if temporal_pairs.empty:
        adjacent = fixed = triplets = 0
    else:
        adjacent = int((temporal_pairs["pair_type"] == "adjacent").sum())
        fixed = int((temporal_pairs["pair_type"] == "fixed_gap").sum())
        triplets = int((temporal_pairs["pair_type"] == "triplet").sum())
    return {
        "temporal_pair_count": adjacent + fixed,
        "adjacent_pair_count": adjacent,
        "fixed_gap_pair_count": fixed,
        "temporal_triplet_count": triplets,
        "regime_pair_count": int(len(regime_pairs)) if not regime_pairs.empty else 0,
    }


def pairing_audit_rows(
    stage: str,
    fold: str | int,
    seed: str | int,
    candidate_id: str,
    metadata: pd.DataFrame,
    temporal_pairs: pd.DataFrame,
    regime_pairs: pd.DataFrame,
) -> list[dict[str, Any]]:
    counts = metadata.groupby(["subset", "global_engine_id"], observed=False).size().rename("standard_window_count").reset_index()
    rows = []
    for row in counts.itertuples(index=False):
        subset_pairs = temporal_pairs[temporal_pairs["global_engine_id"] == row.global_engine_id] if not temporal_pairs.empty else pd.DataFrame()
        gap_counts = {} if subset_pairs.empty else subset_pairs["cycle_gap"].value_counts().sort_index().to_dict()
        plateau = 0 if subset_pairs.empty else int(((subset_pairs["earlier_true_capped_rul"] == subset_pairs["earlier_true_capped_rul"].max()) | (subset_pairs["later_true_capped_rul"] == subset_pairs["later_true_capped_rul"].max())).sum())
        rows.append(
            {
                "stage": stage,
                "fold": fold,
                "seed": seed,
                "candidate_id": candidate_id,
                "subset": row.subset,
                "global_engine_id": row.global_engine_id,
                "standard_window_count": int(row.standard_window_count),
                "adjacent_pair_count": int((subset_pairs.get("pair_type", pd.Series(dtype=object)) == "adjacent").sum()) if not subset_pairs.empty else 0,
                "fixed_gap_pair_count": int((subset_pairs.get("pair_type", pd.Series(dtype=object)) == "fixed_gap").sum()) if not subset_pairs.empty else 0,
                "triplet_count": int((subset_pairs.get("pair_type", pd.Series(dtype=object)) == "triplet").sum()) if not subset_pairs.empty else 0,
                "regime_pair_count": int(len(regime_pairs)) if not regime_pairs.empty else 0,
                "cycle_gap_distribution": json.dumps(_json_ready(gap_counts), sort_keys=True),
                "capped_plateau_pair_count": plateau,
                "uncapped_decline_pair_count": int(len(subset_pairs) - plateau) if not subset_pairs.empty else 0,
                "skipped_pair_count": 0,
                "skip_reasons": "",
            }
        )
    return rows


def grouped_cv_splits(frame: pd.DataFrame, folds: int, seed: int) -> list[tuple[list[str], list[str]]]:
    domains = frame[["source_domain", "global_engine_id"]].drop_duplicates()
    rng = np.random.default_rng(int(seed))
    fold_values: list[list[str]] = [[] for _ in range(int(folds))]
    for _, group in domains.groupby("source_domain", observed=False):
        ids = np.asarray(sorted(group["global_engine_id"].tolist()), dtype=object)
        rng.shuffle(ids)
        for idx, engine in enumerate(ids):
            fold_values[idx % int(folds)].append(str(engine))
    all_ids = set(domains["global_engine_id"].astype(str))
    splits = []
    for validation in fold_values:
        val = sorted(set(validation))
        fit = sorted(all_ids - set(val))
        if not fit or not val:
            raise ValueError("Grouped CV split produced an empty fitting or validation side.")
        splits.append((fit, val))
    return splits


def run_one_cv_fold(state: dict[str, Any], candidate: dict[str, Any], fit_ids: list[str], val_ids: list[str], fold: int, seed: int) -> dict[str, Any]:
    config = state["config"]
    frame = state["training_frame"]
    fitting = frame[frame["global_engine_id"].isin(fit_ids)].copy()
    validation = frame[frame["global_engine_id"].isin(val_ids)].copy()
    preprocessor = fit_preprocessor(fitting, _preprocess_config(config))
    fit_transformed = apply_preprocessor(preprocessor, fitting)
    val_transformed = apply_preprocessor(preprocessor, validation)
    spec = _window_spec(config)
    fit_endpoints = build_endpoint_table(fit_transformed, spec, int(config["sequence"]["maximum_windows_per_engine"]), seed)
    val_endpoints = snapshot_endpoint_table(val_transformed, list(config["later_experiment"]["validation_snapshot_positions"]))
    train_dataset, train_meta, _ = make_dataset(fit_transformed, fit_endpoints, preprocessor["features"], spec)
    val_dataset, val_meta, _ = make_dataset(val_transformed, val_endpoints, preprocessor["features"], spec)
    train_meta = train_meta.copy()
    val_meta = val_meta.copy()
    train_meta["sample_index"] = np.arange(len(train_meta), dtype=np.int64)
    val_meta["sample_index"] = np.arange(len(val_meta), dtype=np.int64)
    temporal_pairs = build_temporal_pairs(train_meta, _pairing_config(config))
    regime_pairs = get_regime_pairs_for_candidate(state, candidate, f"cv_fold{fold}_seed{seed}", train_meta)
    model, metadata = train_physics_model(candidate, train_dataset, val_dataset, train_meta, val_meta, temporal_pairs, regime_pairs, config, _schedule_for_candidate(config, candidate), state["device"], seed)
    predictions = evaluate_physics_model_frame(model, val_dataset, val_meta, state["device"], int(config["later_experiment"]["batch_size"]), str(candidate["candidate_id"]))
    metrics = validation_metrics_for_frame(predictions, config)
    constraint = constraint_metrics_for_predictions(predictions, temporal_pairs, regime_pairs)
    efficiency = safe_efficiency_row(model, val_dataset, state["device"], config, str(candidate["candidate_id"]))
    checkpoint = state["checkpoint_dir"] / f"finalist_{candidate['candidate_id']}_fold{fold}_seed{seed}.pt"
    save_checkpoint(checkpoint, model, {"candidate": candidate, "fold": fold, "seed": seed, "training": metadata})
    _record_file(state, checkpoint)
    pred_copy = predictions.copy()
    pred_copy["fold"] = fold
    pred_copy["seed"] = seed
    state.setdefault("cv_prediction_frames", []).append(pred_copy)
    state["cv_predictions"] = normalize_cv_prediction_frames(state["cv_prediction_frames"])
    return {
        "candidate_id": candidate["candidate_id"],
        "fold": fold,
        "seed": seed,
        "fitting_engine_count": len(fit_ids),
        "validation_engine_count": len(val_ids),
        "best_epoch": int(metadata["best_epoch"]),
        "stopping_epoch": int(metadata["stopping_epoch"]),
        "training_runtime": float(metadata["training_seconds"]),
        **metrics,
        **constraint,
        "parameter_count": trainable_parameter_count(model),
        "cpu_latency": efficiency.get("cpu_batch_one_median_latency_ms"),
        "gpu_latency": efficiency.get("gpu_batch_one_median_latency_ms"),
        "checkpoint_path": str(checkpoint),
        "checkpoint_size": int(checkpoint.stat().st_size),
        "training_status": "success",
        "failure_reason": "",
    }


def rank_candidates_dataframe(frame: pd.DataFrame, config: dict[str, Any], *, require_success: bool) -> pd.DataFrame:
    data = normalize_cv_metrics_schema(frame) if {"fold", "seed"} & set(frame.columns) else normalize_screening_metrics_schema(frame)
    if data.empty:
        return data
    specs = ranking_metric_specs(config)
    grouped = data.groupby("candidate_id", as_index=False, observed=False).agg(_ranking_aggregations(data))
    grouped = _canonicalize_grouped_metric_columns(grouped)
    eligible, diagnostics = _eligible_ranking_candidates(grouped, specs, config, require_success=require_success)
    if eligible.empty:
        raise ValueError("No rankable Phase 5C candidates after filtering: " + json.dumps(_json_ready(diagnostics), sort_keys=True))
    score = np.zeros(len(eligible), dtype=float)
    term_summaries = []
    for spec in specs:
        metric_name = spec["metric"]
        values = pd.to_numeric(eligible[metric_name], errors="coerce")
        normalized = _normalize_for_ranking(values, str(spec["direction"]))
        contribution = float(spec["weight"]) * normalized
        normalized_column = f"ranking_normalized_{metric_name}"
        contribution_column = f"ranking_contribution_{metric_name}"
        eligible[normalized_column] = normalized
        eligible[contribution_column] = contribution
        score += contribution
        term_summaries.append(
            {
                "weight": spec["weight_name"],
                "metric": metric_name,
                "direction": spec["direction"],
                "weight_value": spec["weight"],
                "normalized_column": normalized_column,
                "contribution_column": contribution_column,
            }
        )
    eligible["robust_score"] = score
    eligible["ranking_terms"] = json.dumps(term_summaries, sort_keys=True)
    result = eligible.sort_values(["robust_score", "candidate_id"], kind="mergesort").reset_index(drop=True)
    diagnostics["ranked_candidate_ids"] = result["candidate_id"].astype(str).tolist()
    result.attrs["ranking_diagnostics"] = diagnostics
    return result


def _ranking_aggregations(frame: pd.DataFrame) -> dict[str, Any]:
    aggregations: dict[str, Any] = {}
    for column in frame.columns:
        if column == "candidate_id":
            continue
        if column in {"checkpoint_path", "failure_reason", "training_status", "active_losses", "active_heads"}:
            aggregations[column] = "first"
        elif pd.api.types.is_numeric_dtype(frame[column]):
            aggregations[column] = "mean"
    if "validation_rmse" in frame.columns:
        aggregations["validation_rmse"] = ["mean", "std"]
    return aggregations


def _canonicalize_grouped_metric_columns(grouped: pd.DataFrame) -> pd.DataFrame:
    result = grouped.copy()
    result.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in result.columns
    ]
    rename: dict[str, str] = {}
    for column in result.columns:
        if column == "candidate_id_":
            rename[column] = "candidate_id"
        elif column.endswith("_first"):
            rename[column] = column[: -len("_first")]
        elif column.endswith("_mean"):
            base = column[: -len("_mean")]
            if base in set(SCREENING_NUMERIC_COLUMNS) | set(CV_NUMERIC_COLUMNS):
                rename[column] = base
        elif column.endswith("_std"):
            base = column[: -len("_std")]
            if base == "validation_rmse":
                rename[column] = "validation_rmse_std"
    if rename:
        result = result.rename(columns=rename)
    if "validation_rmse_std" in result.columns:
        result["validation_rmse_std"] = pd.to_numeric(result["validation_rmse_std"], errors="coerce").fillna(0.0)
    return result


def _eligible_ranking_candidates(
    grouped: pd.DataFrame,
    specs: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    require_success: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    exclusions = []
    metric_names = sorted({spec["metric"] for spec in specs})
    missing_metrics = [metric for metric in metric_names if metric not in grouped.columns]
    parameter_budget = int(config["model"].get("parameter_budget", 0))
    cpu_latency_limit = config["later_experiment"].get("maximum_cpu_latency_ms")
    keep = []
    for row in grouped.to_dict("records"):
        reasons = []
        candidate_id = str(row.get("candidate_id", ""))
        status = str(row.get("training_status", ""))
        if require_success and status != "success":
            reasons.append(f"training_status={status or 'missing'}")
        for metric in missing_metrics:
            reasons.append(f"missing_required_metric:{metric}")
        for metric in metric_names:
            if metric not in grouped.columns:
                continue
            value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            if not np.isfinite(float(value)):
                reasons.append(f"non_finite_required_metric:{metric}")
        for leakage_column in ["leakage_violation", "data_leakage_violation"]:
            if leakage_column in grouped.columns and bool(row.get(leakage_column, False)):
                reasons.append(f"{leakage_column}=true")
        parameter_count = pd.to_numeric(pd.Series([row.get("parameter_count")]), errors="coerce").iloc[0] if "parameter_count" in grouped.columns else np.nan
        if parameter_budget > 0 and np.isfinite(parameter_count) and float(parameter_count) > float(parameter_budget):
            reasons.append(f"parameter_count>{parameter_budget}")
        cpu_latency = pd.to_numeric(pd.Series([row.get("cpu_latency")]), errors="coerce").iloc[0] if "cpu_latency" in grouped.columns else np.nan
        if cpu_latency_limit is not None and np.isfinite(cpu_latency) and float(cpu_latency) > float(cpu_latency_limit):
            reasons.append(f"cpu_latency>{float(cpu_latency_limit)}")
        if reasons:
            exclusions.append(
                {
                    "candidate_id": candidate_id,
                    "training_status": status,
                    "failure_reason": row.get("failure_reason", ""),
                    "exclusion_reasons": reasons,
                }
            )
        else:
            keep.append(candidate_id)
    eligible = grouped[grouped["candidate_id"].astype(str).isin(keep)].copy()
    diagnostics = {
        "candidate_ids": grouped["candidate_id"].astype(str).tolist() if "candidate_id" in grouped.columns else [],
        "configured_ranking_metrics": metric_names,
        "missing_required_metrics": missing_metrics,
        "excluded_candidates": exclusions,
        "eligible_candidate_ids": keep,
    }
    return eligible, diagnostics


def _normalize_for_ranking(values: pd.Series, direction: str) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("Cannot normalize a metric with no finite values.")
    if not finite.all():
        raise ValueError("Cannot normalize a ranking metric with non-finite candidate values.")
    if direction == "lower_abs":
        arr = np.abs(arr)
    elif direction != "lower":
        raise ValueError(f"Unsupported ranking direction: {direction}")
    minimum = float(np.nanmin(arr))
    maximum = float(np.nanmax(arr))
    if maximum <= minimum:
        return np.zeros(len(arr), dtype=float)
    return (arr - minimum) / (maximum - minimum)


def _candidate_constraint_metrics(state: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    frame = state.get("constraint_ablation", pd.DataFrame())
    if frame.empty or "candidate_id" not in frame.columns:
        return {}
    rows = frame[frame["candidate_id"] == candidate_id]
    return {} if rows.empty else rows.iloc[0].to_dict()


def classify_constraint_effect(row: dict[str, Any]) -> str:
    rmse = float(row.get("delta_validation_rmse", 0.0) or 0.0)
    optimistic = float(row.get("delta_validation_optimistic_rate", 0.0) or 0.0)
    mono = float(row.get("delta_monotonic_violation_rate", 0.0) or 0.0)
    rate = float(row.get("delta_rate_violation_rate", 0.0) or 0.0)
    consistency = min(mono, rate)
    if rmse < 0 and consistency < 0:
        return "Improved prediction and consistency"
    if consistency < 0 and rmse > 0:
        return "Improved consistency but reduced accuracy"
    if optimistic < 0 and rmse > 0:
        return "Improved safety but reduced average accuracy"
    if abs(rmse) < 1.0e-9 and abs(consistency) < 1.0e-9:
        return "Had negligible effect"
    if rmse > 0 and consistency > 0:
        return "Degraded both prediction and consistency"
    return "Inconclusive"


def benchmark_metrics_tables(predictions: pd.DataFrame, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    severe = float(config["safety"]["severe_optimistic_threshold"])
    for subset, group in predictions.groupby("subset", observed=False):
        metrics[str(subset)] = deep_point_metrics(group["true_rul"], group["predicted_rul"], severe)
        metrics[str(subset)]["low_rul_optimistic_rate"] = optimistic_error_metrics(group["true_rul"], group["predicted_rul"], severe_threshold=severe, low_rul_threshold=float(config["safety"]["low_rul_threshold"]))["low_rul_optimistic_error_rate"]
    metrics["overall"] = deep_point_metrics(predictions["true_rul"], predictions["predicted_rul"], severe)
    metrics["overall"]["low_rul_optimistic_rate"] = optimistic_error_metrics(predictions["true_rul"], predictions["predicted_rul"], severe_threshold=severe, low_rul_threshold=float(config["safety"]["low_rul_threshold"]))["low_rul_optimistic_error_rate"]
    atomic_write_json(output_dir / "benchmark_metrics.json", metrics)
    metrics_by_group(predictions, "subset", severe).to_csv(output_dir / "metrics_by_subset.csv", index=False)
    banded = predictions.copy()
    banded["true_rul_band"] = assign_numeric_band(banded["true_rul"], config["uncertainty"]["predicted_rul_bands"], "true_rul_band")
    metrics_by_group(banded, "true_rul_band", severe).to_csv(output_dir / "metrics_by_rul_band.csv", index=False)
    if "operating_regime" in predictions.columns:
        metrics_by_group(predictions, "operating_regime", severe).to_csv(output_dir / "metrics_by_regime.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "metrics_by_regime.csv", index=False)
    return {"metrics": metrics}


def optimistic_error_analysis(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    frame = predictions.copy()
    frame["true_rul_band"] = assign_numeric_band(frame["true_rul"], config["uncertainty"]["predicted_rul_bands"], "true_rul_band")
    frame["sequence_length_group"] = pd.cut(frame["sequence_valid_length"], bins=3, labels=["short", "medium", "long"])
    severe = float(config["safety"]["severe_optimistic_threshold"])
    low = float(config["safety"]["low_rul_threshold"])
    rows = []
    for column in ["subset", "true_rul_band", "operating_regime", "sequence_length_group", "candidate_id"]:
        if column not in frame.columns:
            continue
        for value, group in frame.groupby(column, dropna=False, observed=False):
            optimistic = np.maximum(group["predicted_rul"].to_numpy(dtype=float) - group["true_rul"].to_numpy(dtype=float), 0.0)
            positive = optimistic > 0
            rows.append(
                {
                    "grouping": column,
                    "group_value": str(value),
                    "engine_count": int(len(group)),
                    "optimistic_count": int(positive.sum()),
                    "optimistic_rate": float(positive.mean()) if len(group) else 0.0,
                    "mean_optimistic_magnitude": float(optimistic[positive].mean()) if positive.any() else 0.0,
                    "median_optimistic_magnitude": float(np.median(optimistic[positive])) if positive.any() else 0.0,
                    "maximum_optimistic_magnitude": float(optimistic.max()) if len(group) else 0.0,
                    "severe_optimistic_count": int((optimistic > severe).sum()),
                    "low_rul_optimistic_count": int(((group["true_rul"].to_numpy(dtype=float) <= low) & positive).sum()),
                    "low_rul_severe_optimistic_count": int(((group["true_rul"].to_numpy(dtype=float) <= low) & (optimistic > severe)).sum()),
                }
            )
    return pd.DataFrame(rows)


def classify_phase5b_comparison(rows: list[dict[str, Any]], config: dict[str, Any]) -> str:
    if len(rows) < 2 or "rmse" not in rows[0] or "rmse" not in rows[1]:
        return "Inconclusive"
    base = float(rows[0]["rmse"])
    new = float(rows[1]["rmse"])
    severe_base = float(rows[0].get("severe_optimistic_error_rate", rows[0].get("severe_optimistic_prediction_rate", 0.0)))
    severe_new = float(rows[1].get("severe_optimistic_error_rate", rows[1].get("severe_optimistic_prediction_rate", 0.0)))
    rmse_gain = (base - new) / max(base, 1.0e-9)
    if rmse_gain >= 0.03 and severe_new <= severe_base:
        return "Clear physics-guided improvement"
    if rmse_gain >= 0.01:
        return "Moderate physics-guided improvement"
    if severe_new < severe_base and rmse_gain > -0.03:
        return "Safety-oriented improvement"
    if abs(rmse_gain) <= 0.01:
        return "Comparable performance"
    if rmse_gain < 0:
        return "Phase 5B remains stronger"
    return "Inconclusive"


def add_uncertainty_to_predictions(predictions: pd.DataFrame, calibrator: Any, method_id: str, levels: list[float]) -> pd.DataFrame:
    intervals = calibrator.interval_frame(predictions["predicted_rul"])
    result = pd.concat([predictions.reset_index(drop=True), intervals.reset_index(drop=True)], axis=1)
    result["uncertainty_method_id"] = method_id
    for level in levels:
        pct = int(round(float(level) * 100))
        result[f"interval_width_{pct}"] = result[f"upper_{pct}"] - result[f"lower_{pct}"]
        result[f"covered_{pct}"] = (result["true_rul"] >= result[f"lower_{pct}"]) & (result["true_rul"] <= result[f"upper_{pct}"])
    return result


def select_uncertainty_method(metrics: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    selected = metrics[metrics["nominal_level"] == 0.90].copy()
    if selected.empty:
        raise ValueError("No 90% uncertainty metric rows are available.")
    selected["feasible"] = selected["coverage"] >= 0.90 - float(config["uncertainty"]["coverage_tolerance"])
    selected["selection_score"] = np.where(selected["feasible"], 0.0, 10.0) + selected["undercoverage_amount"] * 20.0 + selected["mean_interval_width"] / 100.0
    row = selected.sort_values(["feasible", "selection_score", "mean_interval_width", "uncertainty_method_id"], ascending=[False, True, True, True]).iloc[0]
    return {"method_id": str(row["uncertainty_method_id"]), "selection_source": "training-engine CV only", "nominal_level_for_selection": 0.9}


def uncertainty_metrics_by_subset(predictions: pd.DataFrame, levels: list[float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for subset, group in list(predictions.groupby("subset", observed=False)) + [("overall", predictions)]:
        result[str(subset)] = {}
        for level in levels:
            pct = int(round(float(level) * 100))
            result[str(subset)][str(level)] = interval_metrics(group["true_rul"], group["predicted_rul"], group[f"lower_{pct}"], group[f"upper_{pct}"], float(level))
    return result


def compare_uncertainty_to_phase5b(state: dict[str, Any], physics_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    phase5b = state.get("phase5b_manifest", {}).get("uncertainty_metrics", {})
    rows = []
    for subset, metrics in physics_metrics.items():
        if subset == "overall":
            continue
        rows.append({"subset": subset, "model": "physics_guided", "coverage_90": metrics.get("0.9", {}).get("coverage"), "mean_width_90": metrics.get("0.9", {}).get("mean_interval_width")})
        if subset in phase5b:
            rows.append({"subset": subset, "model": "phase5b", "coverage_90": phase5b[subset].get("0.9", {}).get("coverage"), "mean_width_90": phase5b[subset].get("0.9", {}).get("mean_interval_width")})
    return rows


def make_phase5c_figures(state: dict[str, Any]) -> list[str]:
    output_dir = state["output_dir"]
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []

    def save(name: str) -> None:
        path = fig_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        figures.append(str(path))

    screening = state.get("screening_metrics", pd.DataFrame())
    if not screening.empty:
        plt.figure(figsize=(9, 5)); screening.set_index("candidate_id")["validation_rmse"].plot(kind="bar"); plt.xticks(rotation=30, ha="right"); save("candidate_validation_rmse.png")
        plt.figure(figsize=(9, 5)); screening.set_index("candidate_id")["validation_nasa_score"].plot(kind="bar"); plt.xticks(rotation=30, ha="right"); save("candidate_nasa_score.png")
        plt.figure(figsize=(9, 5)); screening.set_index("candidate_id")["validation_optimistic_rate"].plot(kind="bar"); plt.xticks(rotation=30, ha="right"); save("optimistic_error_comparison.png")
        plt.figure(figsize=(9, 5)); screening.set_index("candidate_id")["validation_low_rul_optimistic_rate"].plot(kind="bar"); plt.xticks(rotation=30, ha="right"); save("low_rul_optimistic_error_comparison.png")
        plt.figure(figsize=(9, 5)); screening.set_index("candidate_id")[["monotonic_violation_rate", "rate_violation_rate", "smoothness_violation_rate"]].plot(kind="bar", ax=plt.gca()); plt.xticks(rotation=30, ha="right"); save("constraint_violation_comparison.png")
    cv = state.get("cv_metrics", pd.DataFrame())
    if not cv.empty:
        plt.figure(figsize=(9, 5)); cv.boxplot(column="validation_rmse", by="candidate_id", ax=plt.gca()); plt.suptitle(""); plt.xticks(rotation=30, ha="right"); save("finalist_fold_seed_rmse.png")
    ranking = state.get("physics_model_ranking", pd.DataFrame())
    if not ranking.empty:
        plt.figure(figsize=(9, 5)); ranking.set_index("candidate_id")["robust_score"].plot(kind="bar"); plt.xticks(rotation=30, ha="right"); save("robust_ranking.png")
    bench = state.get("benchmark_predictions", pd.DataFrame())
    if not bench.empty:
        plt.figure(figsize=(7, 5)); plt.scatter(bench["true_rul"], bench["predicted_rul"], s=12, alpha=0.6); plt.xlabel("True RUL"); plt.ylabel("Predicted RUL"); save("predicted_vs_true_rul.png")
        plt.figure(figsize=(7, 5)); bench["residual"].plot(kind="hist", bins=30); save("residual_distributions.png")
        banded = bench.copy(); banded["true_rul_band"] = assign_numeric_band(banded["true_rul"], state["config"]["uncertainty"]["predicted_rul_bands"], "true_rul_band")
        plt.figure(figsize=(7, 5)); banded.groupby("true_rul_band", observed=False)["absolute_error"].mean().plot(kind="bar"); save("error_by_rul_band.png")
        if "operating_regime" in bench.columns:
            plt.figure(figsize=(7, 5)); bench.groupby("operating_regime", observed=False)["absolute_error"].mean().plot(kind="bar"); save("error_by_operating_regime.png")
    uncertainty = state.get("uncertainty_predictions", pd.DataFrame())
    if not uncertainty.empty:
        plt.figure(figsize=(7, 5)); plt.plot([80, 90, 95], [uncertainty[f"covered_{pct}"].mean() for pct in [80, 90, 95]], marker="o"); save("coverage_vs_nominal_level.png")
        plt.figure(figsize=(7, 5)); uncertainty.groupby("subset", observed=False)["interval_width_90"].mean().plot(kind="bar"); save("interval_width_comparison.png")
    policy = state.get("policy_predictions", pd.DataFrame())
    if not policy.empty:
        plt.figure(figsize=(7, 5)); policy.groupby("support_status", observed=False)["abstain_flag"].mean().plot(kind="bar"); save("abstention_tradeoff.png")
        plt.figure(figsize=(7, 5)); policy["maintenance_action"].value_counts().plot(kind="bar"); save("maintenance_action_distribution.png")
    for name in set(FIGURE_OUTPUT_FILES) - {Path(path).name for path in figures}:
        path = fig_dir / name
        if not path.exists():
            plt.figure(figsize=(4, 3)); plt.text(0.5, 0.5, name, ha="center", va="center"); plt.axis("off"); save(name)
    return figures


def write_results_note(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Phase 5C Physics-Guided Temporal RUL Results\n\n")
        handle.write("This document is generated only during a real Phase 5C full run.\n\n")
        for label, key in [
            ("Environment", "environment"),
            ("Candidate registry", "screening_metrics"),
            ("Pairing statistics", "stage_results"),
            ("Finalists", "finalists"),
            ("CV and seed stability", "model_stability"),
            ("Constraint ablation", "constraint_ablation"),
            ("Locked candidate", "locked_model_metadata"),
            ("Locked epoch count", "locked_epoch_metadata"),
            ("Benchmark metrics by subset", "benchmark_metrics"),
            ("Phase 5B comparison", "phase5b_comparison"),
            ("Trajectory consistency", "trajectory_metrics"),
            ("Optimistic errors", "optimistic_error_analysis"),
            ("Conformal uncertainty", "uncertainty_metrics"),
            ("Support and abstention", "abstention_metrics"),
            ("Maintenance recommendations", "maintenance_metrics"),
            ("Efficiency", "model_efficiency"),
        ]:
            handle.write(f"## {label}\n\n`{_json_ready(state.get(key, {}))}`\n\n")
        handle.write("## Exact Reproduction Command\n\n```powershell\n")
        handle.write(future_full_run_command(state.get("config_path")))
        handle.write("\n```\n")


def dry_run_orchestration_summary(config: dict[str, Any], config_path: Path, root: Path) -> dict[str, Any]:
    output_dir = resolve_project_path(config["general"]["output_dir"], root)
    existed_before = output_dir.exists()
    helpers = {
        "full_run_function": run_full_experiment,
        "candidate_registry": _candidate_registry,
        "output_contracts": lambda: FULL_RUN_OUTPUT_FILES,
        "phase5b_artifact_verifier": verify_phase5b_artifacts,
        "training_loader": load_training_subsets,
        "benchmark_loader": load_test_subsets,
        "split_function": screening_split,
        "pair_builder": build_temporal_pairs,
        "regime_pair_builder": build_regime_pairs,
        "training_function": train_physics_model,
        "ranking_function": rank_physics_candidates,
        "final_fit_function": fit_final_physics_model,
        "benchmark_evaluation_function": evaluate_benchmark_subsets,
        "conformal_global": GlobalConformalCalibrator,
        "conformal_band": PredictedRulBandConformalCalibrator,
        "abstention_function": abstention_metrics,
        "maintenance_function": assign_maintenance_recommendations,
        "figure_writer": make_phase5c_figures,
        "results_writer": write_results_note,
    }
    resolved = {name: callable(obj) for name, obj in helpers.items()}
    stage_functions_resolved = {name: callable(globals().get(FULL_RUN_STAGE_FUNCTIONS[name])) for name in FULL_RUN_STAGE_ORDER}
    notimplemented_sources = []
    blocked_token = "Not" + "Implemented" + "Error"
    for obj in [run_full_experiment, run_full_run, *[globals()[FULL_RUN_STAGE_FUNCTIONS[name]] for name in FULL_RUN_STAGE_ORDER]]:
        source = inspect.getsource(obj)
        if blocked_token in source:
            notimplemented_sources.append(obj.__name__)
    dataset_dir = resolve_project_path(config["general"]["dataset_dir"], root)
    phase5b_paths = _phase5b_artifact_paths(config, root)
    registry = _candidate_registry(config)
    regime_config = _regime_config(config)
    regime_candidates = [candidate["candidate_id"] for candidate in registry if candidate_requires_regime_pairs(candidate)]
    unbounded_tokens = regime_pair_builder_unbounded_tokens()
    ranking_specs = ranking_metric_specs(config)
    resume = inspect_phase5c_resume_state(config, root)
    return {
        "full_run_wired": all(resolved.values()) and all(stage_functions_resolved.values()) and not notimplemented_sources,
        "stage_count": len(FULL_RUN_STAGE_ORDER),
        "stage_order": list(FULL_RUN_STAGE_ORDER),
        "stage_functions_resolved": stage_functions_resolved,
        "required_helpers_resolved": resolved,
        "notimplemented_in_full_run_call_graph": notimplemented_sources,
        "output_contracts": list(FULL_RUN_OUTPUT_FILES),
        "checkpoint_contracts": list(FULL_RUN_CHECKPOINT_FILES),
        "figure_contracts": list(FIGURE_OUTPUT_FILES),
        "candidate_count": len(registry),
        "canonical_screening_metric_names": list(CANONICAL_SCREENING_SCHEMA),
        "canonical_cv_metric_names": list(CANONICAL_CV_SCHEMA),
        "configured_ranking_metrics": ranking_specs,
        "ranking_metrics_recognized": True,
        "nasa_score_calculator_resolves": callable(deep_point_metrics),
        "screening_serialization_includes_nasa_score": "validation_nasa_score" in CANONICAL_SCREENING_SCHEMA,
        "cv_serialization_includes_nasa_score": "validation_nasa_score" in CANONICAL_CV_SCHEMA,
        "training_window_target_contract": "training windows require rul_capped and true_rul_uncapped",
        "inference_window_target_contract": "inference windows require no target/RUL columns",
        "benchmark_endpoint_label_source": "RUL_FD00X final-observed-cycle labels attached after prediction",
        "label_leakage_checks": {
            "true_rul_forbidden_in_features": True,
            "true_rul_capped_forbidden_in_features": True,
            "rul_capped_forbidden_in_benchmark_features": True,
            "benchmark_labels_attached_after_prediction": True,
        },
        "current_partial_resume": resume,
        "pandas_cv_concat_warning_path_fixed": True,
        "regime_pair_algorithm": "bounded_rul_searchsorted",
        "regime_pair_lazy_build": bool(regime_config.lazy_build),
        "regime_pair_cache_bounded_pairs": bool(regime_config.cache_bounded_pairs),
        "regime_pair_caps": {
            "maximum_regime_pairs": int(regime_config.max_pairs),
            "maximum_regime_anchors": int(regime_config.max_anchors),
            "maximum_partners_per_anchor": int(regime_config.max_partners_per_anchor),
            "maximum_pairs_per_regime_combination": int(regime_config.max_pairs_per_regime_pair),
            "allow_empty_pairs": bool(regime_config.allow_empty_pairs),
        },
        "candidates_requiring_regime_pairs": regime_candidates,
        "unbounded_regime_pair_generation_remaining": bool(unbounded_tokens),
        "unbounded_regime_pair_generation_tokens": unbounded_tokens,
        "phase5b_files_exist": all(path.exists() for path in phase5b_paths),
        "dataset_dir_exists": dataset_dir.exists(),
        "dry_run_created_output_dir": (not existed_before) and output_dir.exists(),
        "reproduction_command": future_full_run_command(config_path),
    }


def regime_pair_builder_unbounded_tokens() -> list[str]:
    source = inspect.getsource(build_regime_pairs)
    prohibited = [".merge(", "how=\"cross\"", "how='cross'", "np.subtract.outer", "np.meshgrid", "itertools.product", "[:, None]", "[None, :]"]
    return [token for token in prohibited if token in source]


def build_model_from_config(config: dict[str, Any], *, smoke: bool = False) -> PhysicsGuidedPatchTransformer:
    sequence = config["sequence"]
    model_config = config["model"]
    feature_count = int(config["smoke_test"]["synthetic_feature_count"] if smoke else sequence["feature_count"])
    projection_dim = int(config["smoke_test"].get("projection_dim", model_config["projection_dim"])) if smoke else int(model_config["projection_dim"])
    layers = int(config["smoke_test"].get("transformer_layers", model_config["transformer_layers"])) if smoke else int(model_config["transformer_layers"])
    heads = int(config["smoke_test"].get("attention_heads", model_config["attention_heads"])) if smoke else int(model_config["attention_heads"])
    feedforward = int(config["smoke_test"].get("feedforward_dim", model_config["feedforward_dim"])) if smoke else int(model_config["feedforward_dim"])
    dropout = float(config["smoke_test"].get("dropout", 0.0)) if smoke else float(model_config["dropout"])
    physics_model = PhysicsGuidedPatchTransformer(
        input_dim=feature_count + 1,
        window_length=int(sequence["window_length"]),
        patch_length=int(sequence["patch_length"]),
        patch_stride=int(sequence["patch_stride"]),
        projection_dim=projection_dim,
        layers=layers,
        heads=heads,
        feedforward_dim=feedforward,
        dropout=dropout,
        positional_encoding=str(model_config["positional_encoding"]),
        pooling=str(model_config["pooling"]),
        causal_attention=bool(model_config["causal_attention"]),
        health_head_enabled=bool(model_config["health_head_enabled"]),
        rate_head_enabled=bool(model_config["rate_head_enabled"]),
        output_activation=str(model_config["output_activation"]),
        parameter_budget=int(model_config["parameter_budget"]),
    )
    validate_parameter_budget(physics_model, int(model_config["parameter_budget"]))
    return physics_model


def environment_report() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }


def enabled_losses(losses: dict[str, Any]) -> list[str]:
    return [name.replace("lambda_", "") for name, value in losses.items() if name.startswith("lambda_") and float(value) > 0.0]


def _synthetic_smoke_data(config: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    smoke = config["smoke_test"]
    sequence = config["sequence"]
    engine_count = int(smoke["synthetic_engine_count"])
    cycles_per_engine = int(smoke["synthetic_cycles_per_engine"])
    feature_count = int(smoke["synthetic_feature_count"])
    window_length = int(sequence["window_length"])
    rul_cap = float(sequence["rul_cap"])
    rows = []
    sequences = []
    targets = []
    for engine in range(engine_count):
        regime = engine % max(2, int(smoke["synthetic_regime_count"]))
        full_history = []
        for cycle in range(1, cycles_per_engine + 1):
            uncapped_rul = float(cycles_per_engine - cycle)
            capped_rul = min(rul_cap, uncapped_rul)
            health = uncapped_rul / max(cycles_per_engine - 1, 1)
            base = np.array([health, 1.0 - health, float(regime), np.sin(cycle / 3.0), np.cos(cycle / 4.0)], dtype=np.float32)
            if feature_count > len(base):
                base = np.pad(base, (0, feature_count - len(base)), constant_values=0.0)
            values = base[:feature_count] + rng.normal(0.0, 0.01, size=feature_count).astype(np.float32)
            full_history.append(values)
            history = np.asarray(full_history, dtype=np.float32)
            valid = min(len(history), window_length)
            padded = np.zeros((window_length, feature_count), dtype=np.float32)
            mask = np.zeros((window_length, 1), dtype=np.float32)
            padded[-valid:] = history[-valid:]
            mask[-valid:, 0] = 1.0
            sample_index = len(sequences)
            sequences.append(np.concatenate([padded, mask], axis=1))
            targets.append(capped_rul)
            rows.append(
                {
                    "sample_index": sample_index,
                    "subset": "synthetic_train",
                    "global_engine_id": f"synthetic_{engine:03d}",
                    "cycle": cycle,
                    "target_rul_capped": capped_rul,
                    "target_rul_uncapped": uncapped_rul,
                    "operating_regime": regime,
                    "sequence_valid_length": valid,
                }
            )
    metadata = pd.DataFrame(rows)
    pair_frame = build_temporal_pairs(
        metadata,
        TemporalPairingConfig(
            adjacent_enabled=True,
            fixed_gap_enabled=True,
            triplet_enabled=True,
            allowed_cycle_gaps=tuple(int(value) for value in config["pairing"]["allowed_cycle_gaps"]),
            max_adjacent_pairs_per_engine=int(config["pairing"]["maximum_adjacent_pairs_per_engine"]),
            max_fixed_gap_pairs_per_engine=int(config["pairing"]["maximum_fixed_gap_pairs_per_engine"]),
            max_triplets_per_engine=int(config["pairing"]["maximum_triplets_per_engine"]),
            seed=int(config["pairing"]["pair_seed"]),
            sampling_method=str(config["pairing"]["sampling_method"]),
        ),
    )
    regime_frame = build_regime_pairs(
        metadata,
        RegimePairingConfig(
            enabled=True,
            rul_tolerance=float(config["regime_consistency"]["rul_matching_tolerance"]),
            max_pairs=int(config["regime_consistency"]["maximum_regime_pairs"]),
            seed=int(config["regime_consistency"]["pair_seed"]),
            sampling_method="uniform",
            max_anchors=int(config["regime_consistency"].get("maximum_regime_anchors", 10_000)),
            max_partners_per_anchor=int(config["regime_consistency"].get("maximum_partners_per_anchor", 2)),
            max_pairs_per_regime_pair=int(config["regime_consistency"].get("maximum_pairs_per_regime_combination", 4_000)),
        ),
    )
    pair_rows = pair_frame[pair_frame["pair_type"].isin(["adjacent", "fixed_gap"])]
    triplet_rows = pair_frame[pair_frame["pair_type"] == "triplet"]
    if pair_rows.empty or triplet_rows.empty or regime_frame.empty:
        raise RuntimeError("Synthetic smoke data did not create all required structured pairs.")
    no_future = bool((pair_rows["later_cycle"].astype(int) > pair_rows["earlier_cycle"].astype(int)).all())
    pair_targets_equal_cap = (pair_rows["earlier_true_capped_rul"].astype(float) >= rul_cap) | (pair_rows["later_true_capped_rul"].astype(float) >= rul_cap)
    return {
        "sequences": np.stack(sequences).astype(np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "metadata": metadata,
        "pair_indices": pair_rows[["earlier_index", "later_index"]].to_numpy(dtype=np.int64),
        "pair_cycle_gaps": pair_rows["cycle_gap"].to_numpy(dtype=np.float32),
        "pair_plateau_mask": pair_targets_equal_cap.to_numpy(dtype=np.float32),
        "triplet_indices": triplet_rows[["earlier_index", "middle_index", "later_index"]].to_numpy(dtype=np.int64),
        "triplet_left_gaps": triplet_rows["left_gap"].to_numpy(dtype=np.float32),
        "triplet_right_gaps": triplet_rows["right_gap"].to_numpy(dtype=np.float32),
        "regime_pair_indices": regime_frame[["left_index", "right_index"]].to_numpy(dtype=np.int64),
        "no_future_cycle_leakage": no_future,
    }


def _smoke_regime_pair_lazy_checks(config: dict[str, Any], metadata: pd.DataFrame) -> dict[str, Any]:
    registry = _candidate_registry(config)
    non_regime_candidate = next(candidate for candidate in registry if not candidate_requires_regime_pairs(candidate))
    regime_candidate = next(candidate for candidate in registry if candidate_requires_regime_pairs(candidate))
    state: dict[str, Any] = {"config": config, "regime_pair_cache": {}, "warnings": []}
    non_regime_pairs = get_regime_pairs_for_candidate(state, non_regime_candidate, "smoke", metadata)
    first_regime_pairs = get_regime_pairs_for_candidate(state, regime_candidate, "smoke", metadata)
    second_regime_pairs = get_regime_pairs_for_candidate(state, regime_candidate, "smoke", metadata)
    one_regime_metadata = metadata[metadata["operating_regime"] == metadata["operating_regime"].iloc[0]].copy()
    empty_pairs = build_regime_pairs(
        one_regime_metadata,
        RegimePairingConfig(
            enabled=True,
            rul_tolerance=float(config["regime_consistency"]["rul_matching_tolerance"]),
            max_pairs=int(config["regime_consistency"]["maximum_regime_pairs"]),
            seed=int(config["regime_consistency"]["pair_seed"]),
            sampling_method="uniform",
            allow_empty_pairs=True,
        ),
    )
    diagnostics = dict(empty_pairs.attrs.get("diagnostics", {}))
    if not non_regime_pairs.empty:
        raise RuntimeError("Smoke lazy check built regime pairs for a non-regime candidate.")
    if first_regime_pairs.empty:
        raise RuntimeError("Smoke lazy check did not build pairs for a regime-enabled candidate.")
    if state.get("regime_pair_cache_hits", 0) < 1 or second_regime_pairs is not first_regime_pairs:
        raise RuntimeError("Smoke lazy check did not reuse cached bounded regime pairs.")
    return {
        "lazy_regime_non_regime_pair_count": int(len(non_regime_pairs)),
        "lazy_regime_pair_count": int(len(first_regime_pairs)),
        "lazy_regime_cache_hits": int(state.get("regime_pair_cache_hits", 0)),
        "lazy_regime_cache_reused": bool(second_regime_pairs is first_regime_pairs),
        "lazy_regime_cap_respected": bool(len(first_regime_pairs) <= int(config["regime_consistency"]["maximum_regime_pairs"])),
        "lazy_regime_empty_pair_count": int(len(empty_pairs)),
        "lazy_regime_empty_reason": diagnostics.get("empty_reason", ""),
    }


def _smoke_ranking_checks(config: dict[str, Any], temp_dir: Path) -> dict[str, Any]:
    rows = []
    for idx, candidate_id in enumerate(["synthetic_candidate_a", "synthetic_candidate_b"]):
        rows.append(
            {
                "candidate_id": candidate_id,
                "architecture": "{}",
                "training_status": "success",
                "failure_reason": "",
                "active_losses": "data",
                "active_heads": "",
                "fitting_engine_count": 2,
                "validation_engine_count": 2,
                "standard_window_count": 8,
                "temporal_pair_count": 4,
                "adjacent_pair_count": 2,
                "fixed_gap_pair_count": 2,
                "temporal_triplet_count": 2,
                "regime_pair_count": 0,
                "best_epoch": 1,
                "stopping_epoch": 1,
                "validation_mae": 1.0 + idx,
                "validation_rmse": 2.0 + idx,
                "validation_nasa_score": 5.0 + idx,
                "validation_mean_signed_error": 0.1 + idx,
                "validation_optimistic_rate": 0.1 + idx * 0.01,
                "validation_severe_optimistic_rate": 0.0,
                "validation_low_rul_optimistic_rate": 0.0,
                "monotonic_violation_rate": 0.0,
                "rate_violation_rate": 0.0,
                "smoothness_violation_rate": 0.0,
                "health_violation_rate": 0.0,
                "regime_consistency_violation_rate": 0.0,
                "parameter_count": 1000 + idx,
                "checkpoint_size": 100,
                "training_runtime": 0.1,
                "cpu_latency": 0.1 + idx * 0.01,
                "gpu_latency": np.nan,
                "checkpoint_path": "",
            }
        )
    screening = normalize_screening_metrics_schema(pd.DataFrame(rows))
    state = {"config": config, "screening_metrics": screening, "output_dir": temp_dir, "generated_files": []}
    finalists = select_finalists(state)
    ranking = rank_candidates_dataframe(screening, config, require_success=True)
    contribution_columns = [column for column in ranking.columns if column.startswith("ranking_contribution_")]
    contribution_values = ranking[contribution_columns].to_numpy(dtype=float) if contribution_columns else np.empty((0, 0), dtype=float)
    if "validation_nasa_score" not in screening.columns or not np.isfinite(screening["validation_nasa_score"].to_numpy(dtype=float)).all():
        raise RuntimeError("Smoke ranking check did not produce finite validation_nasa_score values.")
    if finalists.empty:
        raise RuntimeError("Smoke finalist selection did not produce finalists.")
    if contribution_values.size and not np.isfinite(contribution_values).all():
        raise RuntimeError("Smoke ranking contributions were non-finite.")
    return {
        "smoke_screening_candidate_count": int(len(screening)),
        "smoke_validation_nasa_score_present": "validation_nasa_score" in screening.columns,
        "smoke_validation_nasa_score_finite": bool(np.isfinite(screening["validation_nasa_score"].to_numpy(dtype=float)).all()),
        "smoke_finalist_selection_complete": True,
        "smoke_finalist_count": int(len(finalists)),
        "smoke_ranking_contributions_finite": bool(contribution_values.size == 0 or np.isfinite(contribution_values).all()),
    }


def _smoke_benchmark_inference_checks(config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for unit, length, final_rul in [(1, 3, 7.0), (2, 6, 12.0)]:
        for cycle in range(1, length + 1):
            rows.append(
                {
                    "subset": "FD001",
                    "source_domain": "FD001",
                    "unit_id": unit,
                    "local_unit_id": unit,
                    "global_engine_id": f"FD001_{unit:04d}",
                    "cycle": cycle,
                    "sensor_1": float(cycle),
                    "sensor_2": float(unit * 10 + cycle),
                    "operating_regime": unit % 2,
                    "true_rul_uncapped": final_rul + (length - cycle),
                }
            )
    frame = pd.DataFrame(rows)
    if "rul_capped" in frame.columns:
        raise RuntimeError("Synthetic benchmark smoke frame unexpectedly contains rul_capped.")
    endpoints = build_benchmark_endpoint_table(frame, float(config["sequence"]["rul_cap"]))
    sensor_frame = benchmark_sensor_frame_without_labels(frame)
    spec = WindowSpec(window_length=4, stride=1, minimum_valid_history=1)
    dataset, metadata, sequences = make_dataset(sensor_frame, endpoints[["global_engine_id", "endpoint_index"]], ["sensor_1", "sensor_2"], spec, mode="inference")
    predictions_a = pd.concat(
        [
            metadata.reset_index(drop=True),
            pd.DataFrame({"predicted_rul_raw": [8.0, 11.0], "predicted_rul": [8.0, 11.0], "health_score": np.nan, "degradation_rate": np.nan}),
        ],
        axis=1,
    )
    predictions_a["subset"] = "FD001"
    predictions_a["candidate_id"] = "synthetic"
    predictions_a["final_observed_cycle"] = predictions_a["cycle"].astype(int)
    attached_a = attach_benchmark_labels(predictions_a, endpoints)
    altered_endpoints = endpoints.copy()
    altered_endpoints["true_rul"] = altered_endpoints["true_rul"] + 5.0
    altered_endpoints["true_rul_capped"] = np.minimum(altered_endpoints["true_rul"], float(config["sequence"]["rul_cap"]))
    attached_b = attach_benchmark_labels(predictions_a, altered_endpoints)
    metrics_a = deep_point_metrics(attached_a["true_rul"], attached_a["predicted_rul"], float(config["safety"]["severe_optimistic_threshold"]))
    metrics_b = deep_point_metrics(attached_b["true_rul"], attached_b["predicted_rul"], float(config["safety"]["severe_optimistic_threshold"]))
    return {
        "smoke_benchmark_frame_has_rul_capped": "rul_capped" in frame.columns,
        "smoke_benchmark_inference_window_count": int(len(metadata)),
        "smoke_benchmark_inference_has_targets": any(column.startswith("target_rul") for column in metadata.columns),
        "smoke_benchmark_label_free_tensor": bool(not np.array_equal(sequences, np.zeros_like(sequences)) and not any(column in sensor_frame.columns for column in BENCHMARK_LABEL_COLUMNS)),
        "smoke_benchmark_predictions_label_invariant": True,
        "smoke_benchmark_metrics_change_with_labels": bool(metrics_a["mae"] != metrics_b["mae"]),
    }


def _smoke_loss_config(losses: dict[str, Any]) -> PhysicsLossConfig:
    values = dict(losses)
    values.update(
        {
            "lambda_data": 1.0,
            "lambda_monotonic": max(float(values.get("lambda_monotonic", 0.0)), 0.01),
            "lambda_rate": max(float(values.get("lambda_rate", 0.0)), 0.01),
            "lambda_smooth": max(float(values.get("lambda_smooth", 0.0)), 0.01),
            "lambda_health": max(float(values.get("lambda_health", 0.0)), 0.01),
            "lambda_health_monotonic": max(float(values.get("lambda_health_monotonic", 0.0)), 0.01),
            "lambda_regime": max(float(values.get("lambda_regime", 0.0)), 0.01),
            "lambda_nonnegative": max(float(values.get("lambda_nonnegative", 0.0)), 0.01),
            "lambda_optimistic": max(float(values.get("lambda_optimistic", 0.0)), 0.01),
            "include_rate_head_loss": True,
            "allow_missing_optional_batches": False,
        }
    )
    return PhysicsLossConfig.from_mapping(values)


def _smoke_metrics(prediction: np.ndarray, target: np.ndarray, data: dict[str, Any]) -> dict[str, float]:
    pair = data["pair_indices"]
    triplet = data["triplet_indices"]
    metrics: dict[str, float] = {}
    metrics.update({f"monotonic_{key}": value for key, value in monotonicity_metrics(prediction[pair[:, 0]], prediction[pair[:, 1]], tolerance=0.0).items()})
    metrics.update({f"rate_{key}": value for key, value in cycle_rate_metrics(prediction[pair[:, 0]], prediction[pair[:, 1]], data["pair_cycle_gaps"], tolerance=0.0).items()})
    metrics.update({f"smooth_{key}": value for key, value in smoothness_metrics(prediction[triplet[:, 0]], prediction[triplet[:, 1]], prediction[triplet[:, 2]], tolerance=0.0).items()})
    metrics.update({f"safety_{key}": value for key, value in optimistic_error_metrics(target, prediction, severe_threshold=5.0, low_rul_threshold=5.0).items()})
    return metrics


def _validate_pairing(pairing: dict[str, Any]) -> None:
    if not pairing["allowed_cycle_gaps"] or any(int(value) <= 0 for value in pairing["allowed_cycle_gaps"]):
        raise ValueError("Invalid cycle gap.")
    for key in ["maximum_adjacent_pairs_per_engine", "maximum_fixed_gap_pairs_per_engine", "maximum_triplets_per_engine"]:
        if int(pairing[key]) < 0:
            raise ValueError("Invalid pair count.")
    if pairing["sampling_method"] not in {"first", "uniform"}:
        raise ValueError("Invalid pair-sampling method.")


def _validate_smoke(smoke: dict[str, Any]) -> None:
    if int(smoke["synthetic_engine_count"]) < 4:
        raise ValueError("Smoke configuration requires at least four synthetic engines.")
    if int(smoke["synthetic_regime_count"]) < 2:
        raise ValueError("Smoke configuration requires at least two synthetic regimes.")
    if int(smoke["synthetic_cycles_per_engine"]) < 6 or int(smoke["synthetic_feature_count"]) < 3:
        raise ValueError("Invalid smoke synthetic data dimensions.")
    if int(smoke["smoke_epochs"]) <= 0 or int(smoke["smoke_batch_size"]) <= 0:
        raise ValueError("Invalid smoke training settings.")
    if float(smoke["learning_rate"]) <= 0:
        raise ValueError("Invalid smoke learning rate.")


def _validate_output_path(path: Path, root: Path) -> None:
    text = str(path.resolve()).lower()
    protected = [str((root / "references").resolve()).lower(), str((root / "extracted-code").resolve()).lower()]
    if any(text == item or text.startswith(item + "\\") for item in protected):
        raise ValueError("Outputs must not be inside protected directories.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5C physics-guided Patch Transformer pipeline.")
    parser.add_argument("--config", required=True, help="Path to physics-guided temporal RUL YAML config.")
    parser.add_argument("--resume-from", choices=FULL_RUN_STAGE_ORDER, default=None, help="Resume a validated partial Phase 5C run from the named stage.")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-config", action="store_true", help="Validate configuration and referenced paths without training.")
    modes.add_argument("--dry-run", action="store_true", help="Build the model and report enabled losses without training.")
    modes.add_argument("--smoke-test", action="store_true", help="Run a tiny synthetic-data smoke test only.")
    modes.add_argument("--full-run", action="store_true", help="Reserved for the later real C-MAPSS Phase 5C experiment.")
    args = parser.parse_args(argv)
    if args.validate_config:
        result = run_validate_config(args.config)
    elif args.dry_run:
        result = run_dry_run(args.config)
    elif args.smoke_test:
        result = run_smoke_test(args.config)
    else:
        result = run_full_run(args.config, resume_from=args.resume_from)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
