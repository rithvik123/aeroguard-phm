import json
from pathlib import Path

import pandas as pd
import yaml

from aeroguard.pipelines.optimize_multidomain_temporal_rul import run_pipeline


def _row(unit_id: int, cycle: int, offset: float) -> list[float]:
    degradation = max(cycle - 3, 0)
    settings = [0.02 * unit_id + offset, 0.03 * cycle + offset, 1.0 + 0.01 * ((unit_id + cycle) % 2)]
    sensors = [
        float(sensor + 0.04 * unit_id + 0.02 * cycle + 0.01 * degradation * ((sensor % 4) + 1) + offset)
        for sensor in range(1, 22)
    ]
    return [unit_id, cycle, *settings, *sensors]


def _write_rows(path: Path, rows: list[list[float]]) -> None:
    path.write_text("\n".join(" ".join(str(value) for value in row) for row in rows) + "\n", encoding="utf-8")


def _write_subset(root: Path, subset: str, offset: float) -> None:
    train_rows = [_row(unit, cycle, offset) for unit in [1, 2] for cycle in range(1, 8)]
    test_rows = [_row(1, cycle, offset) for cycle in range(1, 6)]
    _write_rows(root / f"train_{subset}.txt", train_rows)
    _write_rows(root / f"test_{subset}.txt", test_rows)
    (root / f"RUL_{subset}.txt").write_text("2\n", encoding="utf-8")


def _intervals() -> dict:
    return {
        "0.8": {"coverage": 0.8, "mean_interval_width": 8.0, "median_interval_width": 8.0, "undercoverage_amount": 0.0, "mean_interval_score": 8.0},
        "0.9": {"coverage": 0.9, "mean_interval_width": 10.0, "median_interval_width": 10.0, "undercoverage_amount": 0.0, "mean_interval_score": 10.0},
        "0.95": {"coverage": 0.95, "mean_interval_width": 12.0, "median_interval_width": 12.0, "undercoverage_amount": 0.0, "mean_interval_score": 12.0},
    }


def _phase5_artifacts(root: Path) -> tuple[Path, Path, Path]:
    results = root / "phase5"
    checkpoints = root / "phase5_checkpoints"
    results.mkdir()
    checkpoints.mkdir()
    checkpoint = checkpoints / "locked_deep_model.pt"
    checkpoint.write_bytes(b"phase5 checkpoint")
    benchmark_metrics = {
        subset: {"mae": 5.0, "rmse": 6.0, "r2": 0.0, "nasa_score": 10.0, "mean_signed_error": 0.0, "median_absolute_error": 4.0, "p90_absolute_error": 8.0, "optimistic_prediction_rate": 0.5, "severe_optimistic_error_rate": 0.0, "conservative_prediction_rate": 0.5}
        for subset in ["FD001", "FD002", "FD003", "FD004"]
    }
    benchmark_metrics["overall"] = {"mae": 5.0, "rmse": 6.0, "r2": 0.0, "nasa_score": 10.0, "mean_signed_error": 0.0, "median_absolute_error": 4.0, "p90_absolute_error": 8.0, "optimistic_prediction_rate": 0.5, "severe_optimistic_error_rate": 0.0, "conservative_prediction_rate": 0.5}
    uncertainty_metrics = {subset: _intervals() for subset in ["FD001", "FD002", "FD003", "FD004", "overall"]}
    summary = {
        "locked_architecture": "lstm",
        "locked_epoch_count": 4,
        "benchmark_metrics": benchmark_metrics,
        "deep_uncertainty_metrics": uncertainty_metrics,
        "locked_deep_uncertainty_method": {"method_id": "deep_predicted_band_conformal"},
    }
    (results / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (results / "benchmark_metrics.json").write_text(json.dumps(benchmark_metrics), encoding="utf-8")
    pd.DataFrame({"subset": ["FD001"], "global_engine_id": ["FD001_0001"], "true_rul": [2.0], "predicted_rul": [3.0], "absolute_error": [1.0], "abstain_flag": [False]}).to_csv(results / "benchmark_predictions.csv", index=False)
    pd.DataFrame({"subset": ["FD001", "FD002", "FD003", "FD004"], "global_engine_id": ["FD001_0001", "FD002_0001", "FD003_0001", "FD004_0001"], "true_rul": [2.0, 2.0, 2.0, 2.0], "predicted_rul": [3.0, 3.0, 3.0, 3.0], "absolute_error": [1.0, 1.0, 1.0, 1.0], "abstain_flag": [False, False, False, False]}).to_csv(results / "deep_uncertainty_predictions.csv", index=False)
    (results / "deep_uncertainty_metrics.json").write_text(json.dumps(uncertainty_metrics), encoding="utf-8")
    pd.DataFrame({"model_id": ["lstm"], "parameter_count": [100], "serialized_size_bytes": [1000], "cpu_batch_one_median_latency_ms": [0.5], "gpu_batch_one_median_latency_ms": [None]}).to_csv(results / "model_efficiency.csv", index=False)
    (results / "classical_vs_deep.csv").write_text("subset,model,rmse\nFD001,deep_lstm,6\n", encoding="utf-8")
    (results / "locked_deep_model.json").write_text('{"model_id":"lstm"}', encoding="utf-8")
    return results, checkpoint, results / "phase5.yaml"


def _config(dataset_dir: Path, phase5_results: Path, phase5_checkpoint: Path, phase5_config: Path, output_dir: Path, checkpoint_dir: Path) -> dict:
    schedules = {
        "a": {"learning_rate": 0.01, "max_epochs": 2, "minimum_epochs": 1, "early_stopping_patience": 0, "scheduler": "plateau", "scheduler_factor": 0.5, "scheduler_patience": 0, "min_learning_rate": 1e-6},
    }
    bands = [{"label": "low", "lower": 0, "upper": 3}, {"label": "high", "lower": 3.000001, "upper": None}]
    return {
        "dataset_dir": str(dataset_dir),
        "training_subsets": ["FD001", "FD002", "FD003", "FD004"],
        "benchmark_test_subsets": ["FD001", "FD002", "FD003", "FD004"],
        "phase5_config_path": str(phase5_config),
        "phase5_results_path": str(phase5_results),
        "phase5_checkpoint_path": str(phase5_checkpoint),
        "random_seed": 5,
        "execution_profile": "cpu_safe",
        "device": "cpu",
        "deterministic_algorithms": False,
        "window_length": 4,
        "window_stride": 1,
        "minimum_valid_history": 2,
        "maximum_windows_per_engine": 3,
        "sampling_method": "engine_balanced_uniform",
        "validation_snapshot_positions": [0.5, 1.0],
        "rul_bands": bands,
        "training_target": "rul_capped",
        "rul_cap": 8,
        "healthy_rul_threshold": 8,
        "critical_rul_threshold": 2,
        "include_cycle_as_feature": False,
        "features_to_exclude": [],
        "near_constant_threshold": 0.0,
        "correlation_threshold": 0.999,
        "operating_condition_method": "regime_standardization",
        "number_of_operating_regimes": 2,
        "residualization_ridge_alpha": 1.0,
        "screening_validation_fraction": 0.5,
        "screening_seed": 7,
        "maximum_candidate_count": 3,
        "finalist_count": 2,
        "finalist_cv_folds": 2,
        "finalist_cv_seed": 8,
        "finalist_seeds": [9, 10],
        "maximum_seed_run_count": 8,
        "execution_limits": {"cpu_safe": {"finalist_count": 2, "finalist_seed_count": 2, "stage_a_epoch_cap": 2, "finalist_epoch_cap": 2, "reason": "smoke"}},
        "training_schedules": schedules,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "loss": "mse",
        "gradient_clip_norm": 5.0,
        "batch_size": 8,
        "num_workers": 0,
        "mixed_precision": "false",
        "severe_optimistic_threshold": 5,
        "parameter_budget": {"default": 50000, "absolute_max": 100000},
        "latency_feasibility_threshold_ms": 10,
        "latency_repetitions": 1,
        "robust_selection_weights": {"mean_rmse": 1, "rmse_std": 0.5, "nasa_score": 0.02, "optimistic_rate": 10},
        "improvement_classification": {"clear_min_rmse_reduction_fraction": 0.03, "moderate_min_rmse_reduction_fraction": 0.01, "comparable_abs_rmse_delta_fraction": 0.01, "max_nasa_score_increase_fraction": 0.1, "max_optimistic_rate_increase": 0.1, "max_latency_increase_fraction": 2},
        "transformer_defaults": {"projection_dims": [8], "layer_counts": [1], "head_counts": [2], "feedforward_dims": [16], "dropout_values": [0.0], "positional_encoding_methods": ["sinusoidal"], "pooling_methods": ["mean"], "causal_attention": False},
        "patch_options": {"patch_lengths": [2], "patch_strides": [2]},
        "model_registry": [
            {"model_id": "lstm_small", "architecture": "lstm", "schedule_id": "a", "hidden_dim": 6, "layers": 1, "dropout": 0.0},
            {"model_id": "transformer_small", "architecture": "temporal_transformer", "schedule_id": "a", "projection_dim": 8, "layers": 1, "heads": 2, "feedforward_dim": 16, "dropout": 0.0, "positional_encoding": "sinusoidal", "pooling": "mean", "causal_attention": False},
            {"model_id": "patch_small", "architecture": "patch_transformer", "schedule_id": "a", "patch_length": 2, "patch_stride": 2, "projection_dim": 8, "layers": 1, "heads": 2, "feedforward_dim": 16, "dropout": 0.0, "positional_encoding": "sinusoidal", "pooling": "mean", "causal_attention": False},
        ],
        "nominal_coverage_levels": [0.8, 0.9, 0.95],
        "conformal_methods": ["global_grouped_conformal", "predicted_rul_band_conformal"],
        "predicted_rul_bands": bands,
        "coverage_tolerance": 0.2,
        "support_settings": {"support_percentile_range": [0.01, 0.99], "limited_feature_exceedance": 0.2, "out_feature_exceedance": 0.5, "limited_robust_distance": 4.0, "out_robust_distance": 8.0, "regime_distance_quantile": 0.99},
        "abstention_settings": {"max_feature_exceedance_fraction": 0.5, "max_regime_distance": 10.0, "max_interval_width_ratio": 5.0, "min_plausible_rul": 0, "max_plausible_rul": 50, "abstain_on_quantile_crossing": True, "high_error_threshold": 5},
        "maintenance_thresholds": {"urgent_review_max": 1, "schedule_maintenance_max": 2, "plan_inspection_max": 4},
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "representative_engine_count": 2,
        "plotting": {"scatter_sample_rows": 100},
    }


def test_multidomain_temporal_optimization_smoke(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "cmapss"
    dataset_dir.mkdir()
    for index, subset in enumerate(["FD001", "FD002", "FD003", "FD004"]):
        _write_subset(dataset_dir, subset, index * 0.1)
    phase5_results, phase5_checkpoint, phase5_config = _phase5_artifacts(tmp_path)
    phase5_config.write_text("ok: true\n", encoding="utf-8")
    output_dir = tmp_path / "reports" / "deep_rul_extended"
    checkpoint_dir = tmp_path / "artifacts" / "deep_rul_extended" / "checkpoints"
    config_path = tmp_path / "phase5b.yaml"
    config_path.write_text(yaml.safe_dump(_config(dataset_dir, phase5_results, phase5_checkpoint, phase5_config, output_dir, checkpoint_dir)), encoding="utf-8")

    result = run_pipeline(config_path)

    required = [
        "phase5_benchmark_manifest.json",
        "extended_model_registry.json",
        "screening_metrics.csv",
        "finalist_cross_validation_metrics.csv",
        "model_stability.csv",
        "locked_extended_model.json",
        "benchmark_predictions.csv",
        "uncertainty_predictions.csv",
        "phase5_vs_phase5b.csv",
        "phase5_vs_phase5b_uncertainty.csv",
        "run_summary.json",
    ]
    assert all((output_dir / name).exists() for name in required)
    assert (checkpoint_dir / "locked_extended_model.pt").exists()
    assert any(checkpoint_dir.glob("stage_a_*.pt"))
    assert any((output_dir / "figures").glob("*.png"))
    assert set(pd.read_csv(output_dir / "benchmark_predictions.csv")["subset"]) == {"FD001", "FD002", "FD003", "FD004"}
    assert result["locked_model_id"] in {"lstm_small", "transformer_small", "patch_small"}
    assert result["phase5_benchmark_manifest"]["statement"].endswith("not modified by Phase 5B.")
