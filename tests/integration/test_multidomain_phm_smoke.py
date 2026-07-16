from pathlib import Path

import pandas as pd
import yaml

from aeroguard.pipelines.train_multidomain_phm import run_pipeline


def _row(unit_id: int, cycle: int, offset: float) -> list[float]:
    degradation = max(cycle - 4, 0)
    return [
        unit_id,
        cycle,
        0.05 * unit_id + offset,
        0.03 * cycle + offset,
        1.0 + 0.01 * ((unit_id + cycle) % 3) + offset,
        *[
            float(
                sensor
                + 0.12 * unit_id
                + 0.04 * cycle
                + 0.015 * sensor * degradation
                + 0.001 * sensor * cycle * unit_id
                + offset
            )
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


def _policy(policy_id: str, fusion_method: str = "median") -> dict:
    policy = {
        "policy_id": policy_id,
        "calibration_method": "empirical_percentile",
        "fusion_method": fusion_method,
        "threshold": 0.90,
        "persistence": {"type": "consecutive", "k": 2},
        "hysteresis": {
            "enter_threshold": 0.85,
            "exit_threshold": 0.60,
            "min_enter_duration": 2,
            "min_clear_duration": 2,
        },
        "operational_profile": "balanced",
    }
    if fusion_method == "weighted_mean":
        policy["weights"] = {
            "pca_reconstruction": 0.2,
            "isolation_forest": 0.4,
            "one_class_svm": 0.4,
        }
    return policy


def _config(dataset_dir: Path, output_dir: Path, notes_dir: Path) -> dict:
    return {
        "dataset_dir": str(dataset_dir),
        "training_subsets": ["FD001", "FD002", "FD003"],
        "development_test_subsets": ["FD001", "FD002", "FD003"],
        "external_subset": "FD004",
        "random_seed": 42,
        "include_cycle_as_feature": False,
        "features_to_exclude": [],
        "healthy_rul_threshold": 4,
        "critical_rul_threshold": 2,
        "near_constant_threshold": 0.0,
        "correlation_threshold": 0.999,
        "validation": {"folds": 2, "repeats": 1, "seeds": [7]},
        "candidate_normalization_methods": ["none", "global_standardization"],
        "operating_regime_counts": {"default": 2, "none": 2, "global_standardization": 2},
        "residualization": {"ridge_alpha": 1.0},
        "maximum_method_count": 4,
        "method_registry": [
            {
                "method_id": "none_median",
                "normalization_method": "none",
                "policy": _policy("none_median"),
            },
            {
                "method_id": "global_weighted",
                "normalization_method": "global_standardization",
                "policy": _policy("global_weighted", "weighted_mean"),
            },
        ],
        "utility_weights": {
            "detection_rate": 1.0,
            "critical_region_recall": 0.5,
            "detected_before_30_fraction": 0.1,
            "detected_before_60_fraction": 0.1,
            "false_alarm_engine_rate": 0.5,
            "max_domain_false_alarm_engine_rate": 0.2,
            "missed_engine_rate": 0.2,
            "healthy_region_false_positive_rate": 0.2,
            "fold_variability": 0.1,
            "domain_variability": 0.1,
            "alert_instability": 0.01,
        },
        "feasibility_constraints": {
            "max_mean_false_alarm_engine_rate": 1.0,
            "max_mean_healthy_region_false_positive_rate": 1.0,
            "minimum_mean_detection_rate": 0.0,
            "minimum_mean_critical_region_recall": 0.0,
        },
        "detectors": {
            "pca_reconstruction": {"n_components": 1, "threshold_percentile": 90.0},
            "isolation_forest": {
                "n_estimators": 5,
                "max_samples": "auto",
                "contamination": 0.2,
                "random_state": 42,
                "n_jobs": 1,
            },
            "one_class_svm": {"kernel": "rbf", "nu": 0.2, "gamma": "scale", "max_healthy_training_rows": 20},
        },
        "health_index": {"n_components": 1, "lower_quantile": 0.05, "upper_quantile": 0.95, "clip_scaled": True},
        "smoothing": {"method": "median", "window": 3, "causal": True},
        "score_calibration": {
            "method": "empirical_percentile",
            "lower_quantile": 0.05,
            "upper_quantile": 0.95,
            "epsilon": 1.0e-9,
            "clip": True,
        },
        "operational_alert_thresholds": {
            "monitor_score": 0.50,
            "warning_score": 0.70,
            "critical_score": 0.90,
            "warning_health_index_max": 0.60,
            "critical_health_index_max": 0.40,
        },
        "rul_baseline": {
            "ridge_alpha": 1.0,
            "random_forest": {"n_estimators": 5, "max_depth": 4, "min_samples_leaf": 1, "n_jobs": 1},
        },
        "bootstrap_samples": 20,
        "confidence_level": 0.90,
        "bootstrap_seed": 99,
        "generalization_criteria": {
            "strong": {"fd004_detection_rate": 0.0, "fd004_false_alarm_engine_rate": 1.0, "fd004_missed_engine_rate": 1.0, "fd004_critical_region_recall": 0.0, "fd004_rul_mae": 999.0},
            "moderate": {"fd004_detection_rate": 0.0, "fd004_false_alarm_engine_rate": 1.0, "fd004_missed_engine_rate": 1.0, "fd004_critical_region_recall": 0.0, "fd004_rul_mae": 999.0},
            "weak": {"fd004_detection_rate": 0.0, "fd004_false_alarm_engine_rate": 1.0, "fd004_missed_engine_rate": 1.0, "fd004_critical_region_recall": 0.0, "fd004_rul_mae": 999.0},
        },
        "output_dir": str(output_dir),
        "design_note_path": str(notes_dir / "multidomain_phm_design.md"),
        "results_note_path": str(notes_dir / "multidomain_phm_results.md"),
        "representative_timeline_count": 1,
        "plotting": {"scatter_sample_rows": 100},
    }


def test_multidomain_phm_smoke(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "cmapss"
    output_dir = tmp_path / "reports" / "multidomain_phm"
    notes_dir = tmp_path / "notes"
    dataset_dir.mkdir()
    notes_dir.mkdir()
    for index, subset in enumerate(["FD001", "FD002", "FD003", "FD004"]):
        _write_subset(dataset_dir, subset, offset=float(index) * 0.2)

    config_path = tmp_path / "multidomain_phm.yaml"
    config_path.write_text(yaml.safe_dump(_config(dataset_dir, output_dir, notes_dir)), encoding="utf-8")

    result = run_pipeline(config_path)

    required = [
        output_dir / "method_registry.json",
        output_dir / "validation_method_metrics.csv",
        output_dir / "leave_one_domain_out_metrics.csv",
        output_dir / "method_ranking.csv",
        output_dir / "locked_multidomain_method.json",
        output_dir / "final_fit_metadata.json",
        output_dir / "fd001_development_metrics.json",
        output_dir / "fd002_development_metrics.json",
        output_dir / "fd003_development_metrics.json",
        output_dir / "fd004_external_metrics.json",
        output_dir / "fd004_external_engine_summary.csv",
        output_dir / "fd004_cycle_level_alerts.csv",
        output_dir / "rul_transfer_metrics.json",
        output_dir / "rul_transfer_predictions.csv",
        output_dir / "bootstrap_confidence_intervals.json",
        output_dir / "domain_feature_audit.csv",
        output_dir / "domain_shift_before_after.csv",
        output_dir / "domain_shift_summary.json",
        output_dir / "generalization_conclusion.json",
        output_dir / "run_summary.json",
        notes_dir / "multidomain_phm_design.md",
        notes_dir / "multidomain_phm_results.md",
    ]
    assert all(path.exists() for path in required)
    assert any((output_dir / "figures").glob("*.png"))
    assert any((output_dir / "engine_timelines").glob("*.png"))
    assert result["fd004_external_metrics"]["evaluation_label"] == "FD004 untouched external evaluation"
    assert result["validation_best_row"]["method_id"] in {"none_median", "global_weighted"}

    ranking = pd.read_csv(output_dir / "method_ranking.csv")
    fd004_alerts = pd.read_csv(output_dir / "fd004_cycle_level_alerts.csv")
    assert not ranking.empty
    assert {"global_engine_id", "locked_ensemble_score", "locked_persistent_alarm_state"}.issubset(fd004_alerts.columns)
