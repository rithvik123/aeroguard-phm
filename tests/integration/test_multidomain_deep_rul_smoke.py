from pathlib import Path
import json

import pandas as pd
import yaml

from aeroguard.pipelines.train_multidomain_deep_rul import run_pipeline


def _row(unit_id: int, cycle: int, offset: float) -> list[float]:
    degradation = max(cycle - 3, 0)
    settings = [
        0.03 * unit_id + offset,
        0.02 * cycle + offset,
        1.0 + 0.01 * ((unit_id + cycle) % 3) + offset,
    ]
    sensors = [
        float(
            sensor
            + 0.05 * unit_id
            + 0.02 * cycle
            + 0.01 * degradation * ((sensor % 5) + 1)
            + 0.001 * ((cycle + sensor) % 4)
            + offset
        )
        for sensor in range(1, 22)
    ]
    return [unit_id, cycle, *settings, *sensors]


def _write_rows(path: Path, rows: list[list[float]]) -> None:
    path.write_text("\n".join(" ".join(str(value) for value in row) for row in rows) + "\n", encoding="utf-8")


def _write_subset(root: Path, subset: str, offset: float) -> None:
    train_rows = [_row(unit, cycle, offset) for unit in [1, 2] for cycle in range(1, 7)]
    test_rows = [_row(1, cycle, offset) for cycle in range(1, 6)]
    _write_rows(root / f"train_{subset}.txt", train_rows)
    _write_rows(root / f"test_{subset}.txt", test_rows)
    (root / f"RUL_{subset}.txt").write_text("2\n", encoding="utf-8")


def _intervals() -> dict:
    return {
        "0.8": {"coverage": 0.8, "mean_interval_width": 10.0, "median_interval_width": 10.0, "undercoverage_amount": 0.0, "mean_interval_score": 10.0},
        "0.9": {"coverage": 0.9, "mean_interval_width": 12.0, "median_interval_width": 12.0, "undercoverage_amount": 0.0, "mean_interval_score": 12.0},
        "0.95": {"coverage": 0.95, "mean_interval_width": 14.0, "median_interval_width": 14.0, "undercoverage_amount": 0.0, "mean_interval_score": 14.0},
    }


def _phase_artifacts(phase3: Path, phase4: Path) -> None:
    phase3.mkdir(parents=True)
    phase4.mkdir(parents=True)
    (phase3 / "run_summary.json").write_text('{"phase":"3"}', encoding="utf-8")
    (phase3 / "fd004_external_metrics.json").write_text('{"mae": 20.0}', encoding="utf-8")
    (phase3 / "rul_transfer_metrics.json").write_text('{"FD004": {"mae": 20.0}}', encoding="utf-8")
    for path in [phase3 / "phase3.yaml", phase4 / "phase4.yaml", phase4 / "uncertainty_predictions.csv", phase4 / "calibration_conclusion.json", phase4 / "bootstrap_confidence_intervals.json"]:
        path.write_text("ok: true\n" if path.suffix == ".yaml" else "{}\n", encoding="utf-8")
    summary = {
        "locked_point_model": "classical_random_forest",
        "locked_uncertainty_method": {"method_id": "rf_predicted_band_conformal"},
        "test_metrics_by_subset": {
            subset: {
                "point": {"mae": 20.0, "rmse": 25.0, "r2": 0.0, "nasa_score": 1.0},
                "intervals": _intervals(),
            }
            for subset in ["FD001", "FD002", "FD003", "FD004"]
        },
        "abstention_metrics": {subset: {"abstention_rate": 0.0} for subset in ["FD001", "FD002", "FD003", "FD004"]},
    }
    (phase4 / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _config(dataset_dir: Path, phase3: Path, phase4: Path, output_dir: Path, checkpoint_dir: Path, notes_dir: Path) -> dict:
    registry = [
        {"model_id": "sequence_mlp", "architecture": "sequence_mlp", "hidden_dim": 6, "dropout": 0.0},
        {"model_id": "cnn1d", "architecture": "cnn1d", "hidden_dim": 6, "dropout": 0.0, "kernel_size": 3},
        {"model_id": "lstm", "architecture": "lstm", "hidden_dim": 6, "layers": 1, "dropout": 0.0},
        {"model_id": "gru", "architecture": "gru", "hidden_dim": 6, "layers": 1, "dropout": 0.0},
        {"model_id": "tcn", "architecture": "tcn", "hidden_dim": 6, "dropout": 0.0, "kernel_size": 3, "dilations": [1]},
        {"model_id": "cnn_lstm", "architecture": "cnn_lstm", "hidden_dim": 6, "layers": 1, "dropout": 0.0, "kernel_size": 3},
    ]
    bands = [
        {"label": "low", "lower": 0, "upper": 3},
        {"label": "high", "lower": 3.000001, "upper": None},
    ]
    return {
        "dataset_dir": str(dataset_dir),
        "training_subsets": ["FD001", "FD002", "FD003", "FD004"],
        "benchmark_test_subsets": ["FD001", "FD002", "FD003", "FD004"],
        "phase3_config_path": str(phase3 / "phase3.yaml"),
        "phase4_config_path": str(phase4 / "phase4.yaml"),
        "phase3_results_path": str(phase3),
        "phase4_results_path": str(phase4),
        "random_seed": 12,
        "execution_profile": "cpu_safe",
        "device": "cpu",
        "deterministic_algorithms": False,
        "window_length": 4,
        "window_stride": 1,
        "minimum_valid_history": 2,
        "maximum_windows_per_engine": 3,
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
        "screening_seed": 3,
        "finalist_count": 1,
        "finalist_cv_folds": 2,
        "finalist_cv_seed": 4,
        "validation_snapshot_positions": [0.5, 1.0],
        "model_registry": registry,
        "parameter_budget": {"cpu_safe": 50000, "gpu_standard": 50000},
        "optimizer": "adam",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "loss": "mse",
        "gradient_clip_norm": 5.0,
        "screening_max_epochs": 1,
        "finalist_max_epochs": 1,
        "early_stopping_patience": 0,
        "batch_size": 8,
        "num_workers": 0,
        "mixed_precision": "false",
        "deep_improvement_classification": {
            "clear_improvement_min_fd004_mae_reduction": 3.0,
            "moderate_improvement_min_fd004_mae_reduction": 1.0,
            "comparable_abs_fd004_mae_delta": 1.0,
        },
        "nominal_coverage_levels": [0.8, 0.9, 0.95],
        "conformal_methods": ["global_grouped_conformal", "predicted_rul_band_conformal"],
        "predicted_rul_bands": bands,
        "coverage_tolerance": 0.2,
        "support_settings": {
            "support_percentile_range": [0.01, 0.99],
            "limited_feature_exceedance": 0.2,
            "out_feature_exceedance": 0.5,
            "limited_robust_distance": 4.0,
            "out_robust_distance": 8.0,
            "regime_distance_quantile": 0.99,
        },
        "abstention_settings": {
            "max_feature_exceedance_fraction": 0.5,
            "max_regime_distance": 10.0,
            "max_interval_width_ratio": 5.0,
            "min_plausible_rul": 0,
            "max_plausible_rul": 50,
            "abstain_on_quantile_crossing": True,
            "high_error_threshold": 5,
        },
        "maintenance_thresholds": {"urgent_review_max": 1, "schedule_maintenance_max": 2, "plan_inspection_max": 4},
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "design_note_path": str(notes_dir / "multidomain_deep_rul_design.md"),
        "results_note_path": str(notes_dir / "multidomain_deep_rul_results.md"),
        "representative_engine_count": 1,
        "plotting": {"scatter_sample_rows": 100},
        "latency_repetitions": 1,
    }


def test_multidomain_deep_rul_smoke(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "cmapss"
    phase3 = tmp_path / "phase3"
    phase4 = tmp_path / "phase4"
    output_dir = tmp_path / "reports" / "deep_rul"
    checkpoint_dir = tmp_path / "artifacts" / "deep_rul" / "checkpoints"
    notes_dir = tmp_path / "notes"
    dataset_dir.mkdir()
    notes_dir.mkdir()
    for index, subset in enumerate(["FD001", "FD002", "FD003", "FD004"]):
        _write_subset(dataset_dir, subset, offset=float(index) * 0.1)
    _phase_artifacts(phase3, phase4)
    config_path = tmp_path / "multidomain_deep_rul.yaml"
    config_path.write_text(yaml.safe_dump(_config(dataset_dir, phase3, phase4, output_dir, checkpoint_dir, notes_dir)), encoding="utf-8")

    result = run_pipeline(config_path)

    required = [
        "classical_benchmark_manifest.json",
        "deep_model_registry.json",
        "screening_split.json",
        "cross_validation_splits.json",
        "sequence_audit.csv",
        "screening_metrics.csv",
        "finalist_cross_validation_metrics.csv",
        "deep_model_ranking.csv",
        "locked_deep_model.json",
        "benchmark_predictions.csv",
        "benchmark_metrics.json",
        "deep_uncertainty_predictions.csv",
        "deep_uncertainty_metrics.json",
        "classical_vs_deep.csv",
        "classical_vs_deep_uncertainty.csv",
        "model_efficiency.csv",
        "deep_maintenance_recommendations.csv",
        "run_summary.json",
    ]
    assert all((output_dir / name).exists() for name in required)
    assert (checkpoint_dir / "locked_deep_model.pt").exists()
    assert any(checkpoint_dir.glob("screening_*.pt"))
    assert any((output_dir / "figures").glob("*.png"))
    assert any((output_dir / "engine_examples").glob("*.png"))
    assert (notes_dir / "multidomain_deep_rul_design.md").exists()
    assert result["locked_architecture"] in {item["model_id"] for item in _config(dataset_dir, phase3, phase4, output_dir, checkpoint_dir, notes_dir)["model_registry"]}
    assert result["benchmark_test_metadata"].keys() == {"FD001", "FD002", "FD003", "FD004"}

    screening = pd.read_csv(output_dir / "screening_metrics.csv")
    predictions = pd.read_csv(output_dir / "deep_uncertainty_predictions.csv")
    assert set(screening["model_id"]) == {"sequence_mlp", "cnn1d", "lstm", "gru", "tcn", "cnn_lstm"}
    assert {"lower_90", "upper_90", "support_status", "abstain_flag", "maintenance_action"}.issubset(predictions.columns)
    assert set(predictions["subset"]) == {"FD001", "FD002", "FD003", "FD004"}
