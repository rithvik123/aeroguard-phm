from pathlib import Path

import pandas as pd
import yaml

from aeroguard.pipelines.train_multidomain_rul_uncertainty import run_pipeline


def _row(unit_id: int, cycle: int, offset: float) -> list[float]:
    degradation = max(cycle - 4, 0)
    return [
        unit_id,
        cycle,
        0.04 * unit_id + offset,
        0.02 * cycle + offset,
        1.0 + 0.01 * ((unit_id + cycle) % 2) + offset,
        *[
            float(sensor + 0.08 * unit_id + 0.03 * cycle + 0.02 * degradation * sensor + offset)
            for sensor in range(1, 22)
        ],
    ]


def _write_rows(path: Path, rows: list[list[float]]) -> None:
    path.write_text("\n".join(" ".join(str(value) for value in row) for row in rows) + "\n", encoding="utf-8")


def _write_subset(root: Path, subset: str, offset: float) -> None:
    train_rows = [_row(unit, cycle, offset) for unit in [1, 2, 3] for cycle in range(1, 9)]
    test_rows = [_row(unit, cycle, offset) for unit in [1, 2] for cycle in range(1, 7)]
    _write_rows(root / f"train_{subset}.txt", train_rows)
    _write_rows(root / f"test_{subset}.txt", test_rows)
    (root / f"RUL_{subset}.txt").write_text("2\n3\n", encoding="utf-8")


def _phase3_artifacts(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "locked_multidomain_method.json").write_text(
        '{"method_id":"regime_median_0975_3of5","method":{"normalization_method":"regime_standardization"}}',
        encoding="utf-8",
    )
    (root / "fd004_external_metrics.json").write_text(
        '{"engine_level":{"detection_rate":0.5,"false_alarm_engine_rate":0.2},"row_level":{"critical_region_recall":0.7}}',
        encoding="utf-8",
    )
    (root / "rul_transfer_metrics.json").write_text(
        '{"locked_model":"random_forest","test":{"FD004":{"mae":24.0,"rmse":32.0}}}',
        encoding="utf-8",
    )
    (root / "generalization_conclusion.json").write_text('{"classification":"Moderate generalization"}', encoding="utf-8")
    (root / "final_fit_metadata.json").write_text("{}", encoding="utf-8")
    for name in ["method_ranking.csv", "validation_method_metrics.csv", "leave_one_domain_out_metrics.csv"]:
        (root / name).write_text("method_id,value\nx,1\n", encoding="utf-8")


def _config(dataset_dir: Path, phase3_dir: Path, output_dir: Path) -> dict:
    return {
        "dataset_dir": str(dataset_dir),
        "training_subsets": ["FD001", "FD002", "FD003"],
        "development_test_subsets": ["FD001", "FD002", "FD003"],
        "external_benchmark_subset": "FD004",
        "phase3_config_path": str(phase3_dir / "phase3.yaml"),
        "phase3_results_path": str(phase3_dir),
        "random_seed": 42,
        "cross_validation_folds": 2,
        "cross_validation_repeats": 1,
        "cross_validation_seeds": [77],
        "calibration_snapshot_positions": [0.5, 1.0],
        "minimum_snapshots_per_band": 2,
        "nominal_coverage_levels": [0.8, 0.9, 0.95],
        "selection_nominal_level": 0.9,
        "coverage_tolerance": 0.1,
        "point_model_parameters": {
            "normalization_method": "regime_standardization",
            "operating_regime_count": 2,
            "residualization_ridge_alpha": 1.0,
            "rul_cap": 4,
            "healthy_rul_threshold": 4,
            "critical_rul_threshold": 2,
            "near_constant_threshold": 0.0,
            "correlation_threshold": 0.999,
            "features_to_exclude": [],
        },
        "ridge_parameters": {"alpha": 1.0},
        "random_forest_parameters": {"n_estimators": 5, "max_depth": 4, "min_samples_leaf": 1, "random_state": 42, "n_jobs": 1},
        "quantile_gradient_boosting_parameters": {
            "enabled": True,
            "n_estimators": 3,
            "learning_rate": 0.1,
            "max_depth": 1,
            "min_samples_leaf": 1,
            "subsample": 1.0,
        },
        "conformal_methods": {"finite_sample_correction": True, "lower_clip_for_presentation": True},
        "predicted_rul_bands": [
            {"label": "low", "lower": 0, "upper": 3},
            {"label": "high", "lower": 3.000001, "upper": None},
        ],
        "true_rul_bands": [
            {"label": "low", "lower": 0, "upper": 3},
            {"label": "high", "lower": 3.000001, "upper": None},
        ],
        "maximum_method_count": 3,
        "uncertainty_method_registry": [
            {"method_id": "rf_global", "point_model": "random_forest", "interval_method": "global_grouped_conformal", "nominal_levels": [0.8, 0.9, 0.95], "calibration_source": "oof", "grouping_method": "engine", "fallback_behavior": "none", "model_parameters": {}, "random_seed": 42},
            {"method_id": "rf_tree_cal", "point_model": "random_forest", "interval_method": "calibrated_rf_tree_quantile", "nominal_levels": [0.8, 0.9, 0.95], "calibration_source": "oof", "grouping_method": "engine", "fallback_behavior": "expand", "model_parameters": {}, "random_seed": 42},
            {"method_id": "qgb_cqr", "point_model": "random_forest", "interval_method": "conformalized_quantile_gradient_boosting", "nominal_levels": [0.8, 0.9, 0.95], "calibration_source": "oof", "grouping_method": "engine", "fallback_behavior": "exclude", "model_parameters": {}, "random_seed": 42},
        ],
        "support_percentile_range": [0.01, 0.99],
        "support_threshold_candidates": {"limited_feature_exceedance": 0.1, "out_feature_exceedance": 0.5, "limited_robust_distance": 3.0, "out_robust_distance": 8.0},
        "regime_distance_threshold_candidates": {"quantile": 0.99},
        "interval_width_threshold_candidates": {"max_interval_width_ratio": 3.0},
        "abstention_rules": {"max_feature_exceedance_fraction": 0.5, "max_regime_distance": 8.0, "max_interval_width_ratio": 3.0, "min_plausible_rul": 0, "max_plausible_rul": 20, "abstain_on_quantile_crossing": True, "high_error_threshold": 5},
        "maintenance_lower_bound_level": 0.9,
        "maintenance_thresholds": {"urgent_review_max": 1, "schedule_maintenance_max": 2, "plan_inspection_max": 4},
        "bootstrap_samples": 20,
        "bootstrap_seed": 91,
        "confidence_level": 0.9,
        "calibration_classification_criteria": {
            "well_calibrated": {"min_cv_coverage_90": 0, "min_fd004_coverage_90": 0, "max_fd004_mean_width_90": 999, "max_abstention_rate": 1},
            "moderately_calibrated": {"min_cv_coverage_90": 0, "min_fd004_coverage_90": 0, "max_fd004_mean_width_90": 999, "max_abstention_rate": 1},
            "weakly_calibrated": {"min_cv_coverage_90": 0, "min_fd004_coverage_90": 0, "max_fd004_mean_width_90": 999, "max_abstention_rate": 1},
        },
        "output_dir": str(output_dir),
        "representative_engine_count": 1,
        "plotting": {"scatter_sample_rows": 100},
    }


def test_multidomain_rul_uncertainty_smoke(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "cmapss"
    phase3_dir = tmp_path / "phase3"
    output_dir = tmp_path / "reports" / "rul_uncertainty"
    dataset_dir.mkdir()
    for index, subset in enumerate(["FD001", "FD002", "FD003", "FD004"]):
        _write_subset(dataset_dir, subset, float(index) * 0.1)
    _phase3_artifacts(phase3_dir)
    (phase3_dir / "phase3.yaml").write_text("ok: true\n", encoding="utf-8")
    config_path = tmp_path / "multidomain_rul_uncertainty.yaml"
    config_path.write_text(yaml.safe_dump(_config(dataset_dir, phase3_dir, output_dir)), encoding="utf-8")

    result = run_pipeline(config_path)

    required = [
        "phase3_benchmark_manifest.json",
        "cross_validation_splits.json",
        "calibration_snapshots.csv",
        "uncertainty_method_registry.json",
        "cross_validation_uncertainty_metrics.csv",
        "uncertainty_method_ranking.csv",
        "locked_uncertainty_method.json",
        "uncertainty_predictions.csv",
        "calibration_metrics.json",
        "abstention_metrics.json",
        "maintenance_recommendations.csv",
        "bootstrap_confidence_intervals.json",
        "calibration_conclusion.json",
        "run_summary.json",
    ]
    assert all((output_dir / name).exists() for name in required)
    assert any((output_dir / "figures").glob("*.png"))
    assert any((output_dir / "engine_examples").glob("*.png"))
    assert result["cross_validation"]["leakage_report"]["no_engine_overlap"] is True

    predictions = pd.read_csv(output_dir / "uncertainty_predictions.csv")
    assert {"lower_90", "upper_90", "support_status", "abstain_flag", "maintenance_action"}.issubset(predictions.columns)
    assert set(predictions["subset"]) == {"FD001", "FD002", "FD003", "FD004"}
