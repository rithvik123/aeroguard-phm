from pathlib import Path

import yaml

from aeroguard.pipelines.train_fd001_baseline import run_pipeline


def _row(unit_id: int, cycle: int) -> list[float]:
    return [
        unit_id,
        cycle,
        0.01 * cycle,
        0.02 * unit_id,
        1.0,
        *[
            float(sensor * 10 + unit_id * 0.5 + cycle * 0.25)
            for sensor in range(1, 22)
        ],
    ]


def _write_rows(path: Path, rows: list[list[float]]) -> None:
    path.write_text(
        "\n".join(" ".join(str(value) for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_fd001_pipeline_smoke_with_tiny_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "cmapss"
    output_dir = tmp_path / "reports" / "fd001_baseline"
    notes_dir = tmp_path / "notes"
    dataset_dir.mkdir()
    notes_dir.mkdir()

    train_rows = []
    for unit_id in range(1, 5):
        for cycle in range(1, 5):
            train_rows.append(_row(unit_id, cycle))
    test_rows = []
    for unit_id in range(1, 3):
        for cycle in range(1, 4):
            test_rows.append(_row(unit_id, cycle))

    _write_rows(dataset_dir / "train_FD001.txt", train_rows)
    _write_rows(dataset_dir / "test_FD001.txt", test_rows)
    (dataset_dir / "RUL_FD001.txt").write_text("5\n8\n", encoding="utf-8")

    config = {
        "dataset_dir": str(dataset_dir),
        "subset": "FD001",
        "random_seed": 7,
        "validation_fraction": 0.25,
        "target_column": "rul_capped",
        "rul_cap": 3,
        "include_cycle_as_feature": False,
        "features_to_exclude": [],
        "near_constant_threshold": 0.0,
        "correlation_threshold": 0.999,
        "rolling_features_enabled": False,
        "rolling_window_sizes": [3],
        "model_selection_metric": "rmse",
        "clip_predictions_non_negative": True,
        "ridge": {"alpha": 1.0},
        "random_forest": {
            "n_estimators": 2,
            "max_depth": 3,
            "min_samples_leaf": 1,
            "random_state": 7,
            "n_jobs": 1,
        },
        "output_dir": str(output_dir),
        "results_note_path": str(notes_dir / "fd001_baseline_results.md"),
        "plot_sample_engines": 2,
        "sample_sensor_columns": ["sensor_2", "sensor_3"],
    }
    config_path = tmp_path / "fd001_baseline.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_pipeline(config_path)

    assert Path(result["output_files"]["metrics"]).exists()
    assert Path(result["output_files"]["test_predictions"]).exists()
    assert Path(result["output_files"]["feature_audit"]).exists()
    assert Path(result["output_files"]["results_note"]).exists()
    assert len(result["figures"]) == 7
    assert all(Path(path).exists() for path in result["figures"])
    assert set(result["metrics"]["test"]) == {"dummy_median", "ridge", "random_forest"}
