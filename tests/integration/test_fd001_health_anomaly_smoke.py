from pathlib import Path

import yaml

from aeroguard.pipelines.train_fd001_health_anomaly import run_pipeline


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


def test_fd001_health_anomaly_smoke(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "cmapss"
    output_dir = tmp_path / "reports" / "fd001_health_anomaly"
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

    config = {
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
    config_path = tmp_path / "fd001_health_anomaly.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_pipeline(config_path)

    required = [
        output_dir / "feature_set.json",
        output_dir / "row_level_metrics.json",
        output_dir / "engine_level_metrics.json",
        output_dir / "engine_onset_summary.csv",
        output_dir / "cycle_level_scores.csv",
        notes_dir / "fd001_health_anomaly_results.md",
    ]
    assert all(path.exists() for path in required)
    assert any((output_dir / "engine_timelines").glob("*.png"))
    assert result["healthy_training_row_count"] > 0
    assert "smoothed_health_index" in (output_dir / "cycle_level_scores.csv").read_text(encoding="utf-8")
    assert result["validation_selection"]["best_validation_detector_by_f1"] is not None
