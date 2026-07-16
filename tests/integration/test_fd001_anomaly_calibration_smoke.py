from pathlib import Path

import pandas as pd
import yaml

from aeroguard.pipelines.calibrate_fd001_anomaly_alerts import run_pipeline


def _row(unit_id: int, cycle: int) -> list[float]:
    degradation = max(cycle - 4, 0)
    return [
        unit_id,
        cycle,
        0.01 * unit_id,
        0.02 * cycle,
        1.0,
        *[
            float(sensor + unit_id * 0.1 + cycle * 0.05 + degradation * sensor * 0.02)
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


def _phase2b_config(
    dataset_dir: Path,
    baseline_config: Path,
    phase2_config_path: Path,
    output_dir: Path,
    notes_dir: Path,
) -> dict:
    return {
        "dataset_dir": str(dataset_dir),
        "subset": "FD001",
        "baseline_config_path": str(baseline_config),
        "phase2_config_path": str(phase2_config_path),
        "random_seed": 42,
        "validation_fraction": 0.33,
        "healthy_rul_threshold": 4,
        "critical_rul_threshold": 2,
        "calibration_method": "empirical_percentile",
        "calibration_quantiles": {"lower": 0.05, "upper": 0.95},
        "epsilon": 1.0e-9,
        "clip_calibrated_scores": True,
        "candidate_detector_thresholds": [0.80, 0.90],
        "candidate_fusion_methods": ["mean", "weighted_mean", "rank_average"],
        "candidate_fusion_weights": [
            {
                "pca_reconstruction": 0.33,
                "isolation_forest": 0.33,
                "one_class_svm": 0.34,
            },
            {
                "pca_reconstruction": 0.0,
                "isolation_forest": 0.5,
                "one_class_svm": 0.5,
            },
        ],
        "candidate_voting_rules": ["any_one", "at_least_two"],
        "voting_detector_threshold": 0.90,
        "default_persistence": {"consecutive": 2},
        "persistence_base_threshold": 0.85,
        "candidate_consecutive_persistence_values": [2, 3],
        "candidate_k_of_n_rules": [{"k": 2, "n": 3}],
        "score_duration_rules": [{"duration": 2, "threshold": 0.85}],
        "hysteresis": {
            "enter_thresholds": [0.90],
            "exit_thresholds": [0.70],
            "min_enter_duration": 2,
            "min_clear_duration": 2,
        },
        "selected_hysteresis": {
            "enter_threshold": 0.90,
            "exit_threshold": 0.70,
            "min_enter_duration": 2,
            "min_clear_duration": 2,
        },
        "page_hinkley_candidates": [
            {
                "primary_signal": "smoothed_health_index",
                "direction": "decrease",
                "delta": 0.0,
                "threshold": 0.05,
                "min_observations": 3,
                "reset_after_detection": False,
            }
        ],
        "utility_profiles": {
            "safety_first": {
                "detection_reward": 2.0,
                "early_warning_reward": 1.0,
                "missed_engine_penalty": 2.0,
                "false_alarm_engine_penalty": 0.8,
                "healthy_fpr_penalty": 0.8,
                "instability_penalty": 0.4,
                "late_after_critical_penalty": 0.8,
            },
            "balanced": {
                "detection_reward": 1.5,
                "early_warning_reward": 0.7,
                "missed_engine_penalty": 1.2,
                "false_alarm_engine_penalty": 1.0,
                "healthy_fpr_penalty": 1.0,
                "instability_penalty": 0.5,
                "late_after_critical_penalty": 0.5,
            },
            "low_false_alarm": {
                "detection_reward": 1.0,
                "early_warning_reward": 0.4,
                "missed_engine_penalty": 0.8,
                "false_alarm_engine_penalty": 2.0,
                "healthy_fpr_penalty": 2.5,
                "instability_penalty": 0.8,
                "late_after_critical_penalty": 0.4,
            },
        },
        "primary_utility_profile": "balanced",
        "operational_alert_thresholds": {
            "monitor_score": 0.60,
            "warning_score": 0.80,
            "critical_score": 0.95,
            "warning_health_index_max": 0.55,
            "critical_health_index_max": 0.35,
        },
        "use_rul_predictions_in_alerts": False,
        "output_dir": str(output_dir),
        "results_note_path": str(notes_dir / "fd001_anomaly_calibration_results.md"),
        "representative_timeline_count": 1,
        "plotting": {"scatter_sample_rows": 200},
    }


def test_fd001_anomaly_calibration_smoke(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "cmapss"
    phase2_output = tmp_path / "reports" / "fd001_health_anomaly"
    output_dir = tmp_path / "reports" / "fd001_anomaly_calibration"
    notes_dir = tmp_path / "notes"
    baseline_config = tmp_path / "fd001_baseline.yaml"
    dataset_dir.mkdir()
    notes_dir.mkdir()
    baseline_config.write_text("subset: FD001\n", encoding="utf-8")

    train_rows = []
    for unit_id in range(1, 7):
        for cycle in range(1, 9):
            train_rows.append(_row(unit_id, cycle))
    test_rows = []
    for unit_id in range(1, 4):
        for cycle in range(1, 7):
            test_rows.append(_row(unit_id, cycle))
    _write_rows(dataset_dir / "train_FD001.txt", train_rows)
    _write_rows(dataset_dir / "test_FD001.txt", test_rows)
    (dataset_dir / "RUL_FD001.txt").write_text("2\n3\n4\n", encoding="utf-8")

    phase2_config_path = tmp_path / "fd001_health_anomaly.yaml"
    phase2_config_path.write_text(
        yaml.safe_dump(_phase2_config(dataset_dir, baseline_config, phase2_output, notes_dir)),
        encoding="utf-8",
    )
    phase2b_config_path = tmp_path / "fd001_anomaly_calibration.yaml"
    phase2b_config_path.write_text(
        yaml.safe_dump(_phase2b_config(dataset_dir, baseline_config, phase2_config_path, output_dir, notes_dir)),
        encoding="utf-8",
    )

    result = run_pipeline(phase2b_config_path)

    required = [
        output_dir / "score_calibration.json",
        output_dir / "threshold_operating_points.csv",
        output_dir / "ensemble_operating_points.csv",
        output_dir / "persistence_operating_points.csv",
        output_dir / "page_hinkley_operating_points.csv",
        output_dir / "validation_operating_point_ranking.csv",
        output_dir / "selected_operating_points.json",
        output_dir / "validation_metrics.json",
        output_dir / "test_metrics.json",
        output_dir / "engine_alert_summary.csv",
        output_dir / "cycle_level_alerts.csv",
        output_dir / "health_index_generalization.csv",
        output_dir / "run_summary.json",
        notes_dir / "fd001_anomaly_calibration_results.md",
    ]
    assert all(path.exists() for path in required)
    assert any((output_dir / "figures").glob("*.png"))
    assert any((output_dir / "engine_timelines").glob("*.png"))

    calibration_text = (output_dir / "score_calibration.json").read_text(encoding="utf-8")
    assert "higher calibrated score means more anomalous" in calibration_text

    threshold_table = pd.read_csv(output_dir / "threshold_operating_points.csv")
    ensemble_table = pd.read_csv(output_dir / "ensemble_operating_points.csv")
    persistence_table = pd.read_csv(output_dir / "persistence_operating_points.csv")
    cycle_table = pd.read_csv(output_dir / "cycle_level_alerts.csv")

    assert threshold_table["native_reference"].astype(bool).any()
    assert {"score_fusion", "voting"}.issubset(set(ensemble_table["candidate_kind"]))
    assert {"consecutive_2", "2_of_3"}.issubset(set(persistence_table["persistence_rule"]))
    assert {"selected_ensemble_score", "selected_alert_state", "operational_alert_level"}.issubset(cycle_table.columns)
    assert not cycle_table["operational_alert_level"].isna().any()
    assert result["candidate_operating_points_evaluated"] > 0
    assert result["selected_operating_point"]["selected_profile"] == "balanced"
