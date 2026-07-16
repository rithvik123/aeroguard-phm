from pathlib import Path

import pandas as pd
import yaml

from aeroguard.pipelines.validate_fd001_alert_generalization import run_pipeline


def _row(unit_id: int, cycle: int, shift: float = 0.0) -> list[float]:
    degradation = max(cycle - 4, 0)
    return [
        unit_id,
        cycle,
        0.01 * unit_id + shift,
        0.02 * cycle,
        1.0,
        *[
            float(sensor + unit_id * 0.1 + cycle * 0.05 + degradation * sensor * 0.02 + shift)
            for sensor in range(1, 22)
        ],
    ]


def _write_rows(path: Path, rows: list[list[float]]) -> None:
    path.write_text(
        "\n".join(" ".join(str(value) for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _phase2_config(dataset_dir: Path, baseline_config: Path, output_dir: Path, notes_dir: Path) -> dict:
    return {
        "dataset_dir": str(dataset_dir),
        "subset": "FD001",
        "random_seed": 42,
        "validation_fraction": 0.33,
        "baseline_config_path": str(baseline_config),
        "include_cycle_as_feature": False,
        "features_to_exclude": [],
        "healthy_rul_threshold": 4,
        "critical_rul_threshold": 2,
        "near_constant_threshold": 0.0,
        "correlation_threshold": 0.999,
        "health_index": {
            "n_components": 1,
            "lower_quantile": 0.05,
            "upper_quantile": 0.95,
            "clip_scaled": True,
        },
        "smoothing": {"method": "median", "window": 3, "causal": True},
        "pca_reconstruction": {"n_components": 1, "threshold_percentile": 90.0},
        "isolation_forest": {
            "n_estimators": 5,
            "max_samples": "auto",
            "contamination": 0.2,
            "random_state": 42,
            "n_jobs": 1,
        },
        "one_class_svm": {
            "kernel": "rbf",
            "nu": 0.2,
            "gamma": "scale",
            "max_healthy_training_rows": 20,
        },
        "persistence": {
            "window": 2,
            "alarm_state_from_onset": True,
            "require_consecutive_cycles": True,
        },
        "page_hinkley": {
            "primary_signal": "smoothed_health_index",
            "direction": "decrease",
            "delta": 0.0,
            "threshold": 0.05,
            "min_observations": 3,
            "reset_after_detection": False,
        },
        "primary_validation_selection_metric": "pr_auc",
        "output_dir": str(output_dir),
        "results_note_path": str(notes_dir / "fd001_health_anomaly_results.md"),
        "representative_engine_count": 1,
        "representative_timeline_split": "test",
        "figure_settings": {"scatter_sample_rows": 200},
    }


def _profile() -> dict:
    return {
        "detection_rate": 1.5,
        "critical_region_recall": 1.0,
        "missed_engine_rate": 1.2,
        "false_alarm_engine_rate": 1.0,
        "healthy_region_false_positive_rate": 1.0,
        "detected_before_30_fraction": 0.7,
        "detected_before_60_fraction": 0.4,
        "alert_instability": 0.5,
        "utility_variability": 0.5,
    }


def _phase2c_config(dataset_dir: Path, baseline_config: Path, phase2_config_path: Path, output_dir: Path) -> dict:
    policy_base = {
        "calibration_method": "empirical_percentile",
        "persistence": {"type": "consecutive", "k": 2},
        "hysteresis": {"enter_threshold": 0.8, "exit_threshold": 0.6, "min_enter_duration": 1, "min_clear_duration": 1},
        "operational_profile": "balanced",
    }
    return {
        "dataset_dir": str(dataset_dir),
        "fd001_subset": "FD001",
        "fd003_subset": "FD003",
        "baseline_config_path": str(baseline_config),
        "phase2_config_path": str(phase2_config_path),
        "phase2b_config_path": str(phase2_config_path),
        "random_seed": 42,
        "group_cross_validation_folds": 2,
        "group_cross_validation_repeats": 2,
        "group_cross_validation_seeds": [1, 2],
        "healthy_rul_threshold": 4,
        "critical_rul_threshold": 2,
        "score_calibration": {"method": "empirical_percentile", "lower_quantile": 0.05, "upper_quantile": 0.95, "epsilon": 1.0e-9, "clip": True},
        "maximum_policy_count": 4,
        "candidate_policy_registry": [
            {"policy_id": "max", "fusion_method": "max", "threshold": 0.8, **policy_base},
            {
                "policy_id": "iso_svm",
                "fusion_method": "weighted_mean",
                "weights": {"pca_reconstruction": 0.0, "isolation_forest": 0.5, "one_class_svm": 0.5},
                "threshold": 0.8,
                **policy_base,
            },
            {"policy_id": "vote", "fusion_method": "voting", "voting_rule": "at_least_two", "threshold": 0.8, **policy_base},
        ],
        "operational_profiles": {"balanced": _profile(), "safety_first": _profile(), "low_false_alarm": _profile()},
        "primary_operational_profile": "balanced",
        "feasibility_constraints": {
            "max_mean_false_alarm_engine_rate": 1.0,
            "max_mean_healthy_region_false_positive_rate": 1.0,
            "minimum_mean_detection_rate": 0.0,
            "minimum_mean_critical_region_recall": 0.0,
        },
        "utility_variability_penalty": 0.5,
        "alert_unstable_transition_threshold": 2,
        "bootstrap_samples": 20,
        "confidence_level": 0.9,
        "bootstrap_seed": 8,
        "health_index_correlation_categories": {"weak_upper": 0.3, "moderate_upper": 0.6, "strong_lower": 0.6},
        "generalization_classification_criteria": {
            "strong": {"cv_mean_detection_rate": 0.9, "fd003_detection_rate": 0.9},
            "moderate": {"cv_mean_detection_rate": 0.4, "fd003_detection_rate": 0.4},
            "weak": {"cv_mean_detection_rate": 0.0, "fd003_detection_rate": 0.0},
        },
        "operational_alert_thresholds": {
            "monitor_score": 0.5,
            "warning_score": 0.75,
            "critical_score": 0.95,
            "warning_health_index_max": 0.55,
            "critical_health_index_max": 0.35,
        },
        "output_dir": str(output_dir),
        "representative_timeline_count": 1,
        "plotting": {"scatter_sample_rows": 200, "psi_bins": 5},
    }


def test_fd001_alert_generalization_smoke(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "cmapss"
    output_dir = tmp_path / "reports" / "fd001_alert_generalization"
    phase2_output = tmp_path / "reports" / "fd001_health_anomaly"
    notes_dir = tmp_path / "notes"
    baseline_config = tmp_path / "fd001_baseline.yaml"
    dataset_dir.mkdir()
    notes_dir.mkdir()
    baseline_config.write_text("subset: FD001\n", encoding="utf-8")

    fd001_train = [_row(unit_id, cycle) for unit_id in range(1, 7) for cycle in range(1, 9)]
    fd001_test = [_row(unit_id, cycle) for unit_id in range(1, 4) for cycle in range(1, 7)]
    fd003_test = [_row(unit_id, cycle, shift=0.15) for unit_id in range(1, 4) for cycle in range(1, 7)]
    _write_rows(dataset_dir / "train_FD001.txt", fd001_train)
    _write_rows(dataset_dir / "test_FD001.txt", fd001_test)
    _write_rows(dataset_dir / "test_FD003.txt", fd003_test)
    (dataset_dir / "RUL_FD001.txt").write_text("2\n3\n4\n", encoding="utf-8")
    (dataset_dir / "RUL_FD003.txt").write_text("2\n3\n4\n", encoding="utf-8")

    phase2_config_path = tmp_path / "fd001_health_anomaly.yaml"
    phase2_config_path.write_text(
        yaml.safe_dump(_phase2_config(dataset_dir, baseline_config, phase2_output, notes_dir)),
        encoding="utf-8",
    )
    config_path = tmp_path / "fd001_alert_generalization.yaml"
    config_path.write_text(
        yaml.safe_dump(_phase2c_config(dataset_dir, baseline_config, phase2_config_path, output_dir)),
        encoding="utf-8",
    )

    result = run_pipeline(config_path)

    required = [
        "cross_validation_splits.json",
        "candidate_policy_registry.json",
        "cross_validation_fold_metrics.csv",
        "cross_validation_policy_summary.csv",
        "cross_validation_policy_ranking.csv",
        "locked_policy.json",
        "final_fd001_fit_metadata.json",
        "fd001_development_test_metrics.json",
        "fd001_development_test_engine_summary.csv",
        "fd003_external_metrics.json",
        "fd003_external_engine_summary.csv",
        "fd003_cycle_level_alerts.csv",
        "bootstrap_confidence_intervals.json",
        "domain_shift_features.csv",
        "domain_shift_summary.json",
        "health_index_transfer.csv",
        "health_index_transfer_summary.json",
        "generalization_conclusion.json",
        "run_summary.json",
    ]
    assert all((output_dir / name).exists() for name in required)
    assert any((output_dir / "figures").glob("*.png"))
    assert any((output_dir / "engine_timelines").glob("*.png"))
    splits = pd.read_json(output_dir / "cross_validation_splits.json")
    assert "folds" in splits.columns
    cycle = pd.read_csv(output_dir / "fd003_cycle_level_alerts.csv")
    assert {"locked_ensemble_score", "locked_persistent_alarm_state", "locked_operational_alert_level"}.issubset(cycle.columns)
    assert result["candidate_policy_count"] == 3
    assert result["locked_policy"]["test_results_used_for_selection"] is False
