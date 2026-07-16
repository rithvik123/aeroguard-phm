from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

import aeroguard.pipelines.train_physics_guided_temporal_rul as pipeline
from aeroguard.evaluation.metrics import nasa_asymmetric_score


def _config(tmp_path: Path) -> dict:
    return {
        "general": {
            "dataset_dir": str(tmp_path / "data"),
            "training_subsets": ["FD001"],
            "benchmark_test_subsets": ["FD001"],
            "phase5b_config_path": str(tmp_path / "phase5b.yaml"),
            "phase5b_results_path": str(tmp_path / "phase5b"),
            "phase5b_checkpoint_path": str(tmp_path / "phase5b.pt"),
            "output_dir": str(tmp_path / "reports" / "physics_guided_rul"),
            "checkpoint_dir": str(tmp_path / "artifacts" / "physics_guided_rul" / "checkpoints"),
            "overwrite_existing": False,
            "resume_existing": False,
            "random_seed": 7,
            "device": "cpu",
            "deterministic_algorithms": False,
        },
        "sequence": {
            "window_length": 8,
            "window_stride": 1,
            "minimum_valid_history": 2,
            "maximum_windows_per_engine": 3,
            "patch_length": 4,
            "patch_stride": 2,
            "feature_count": 5,
            "rul_cap": 20,
            "training_target": "rul_capped",
            "operating_condition_method": "regime_standardization",
        },
        "model": {
            "projection_dim": 16,
            "transformer_layers": 1,
            "attention_heads": 4,
            "feedforward_dim": 32,
            "dropout": 0.0,
            "positional_encoding": "sinusoidal",
            "pooling": "mean",
            "causal_attention": False,
            "health_head_enabled": True,
            "rate_head_enabled": True,
            "output_activation": "softplus",
            "parameter_budget": 100000,
        },
        "warm_start": {"enabled": False, "checkpoint_path": str(tmp_path / "phase5b.pt"), "load_encoder_only": True, "strict": False},
        "pairing": {
            "adjacent_pair_enabled": True,
            "fixed_gap_pair_enabled": True,
            "triplet_enabled": True,
            "allowed_cycle_gaps": [1, 2],
            "maximum_adjacent_pairs_per_engine": 2,
            "maximum_fixed_gap_pairs_per_engine": 2,
            "maximum_triplets_per_engine": 1,
            "pair_seed": 8,
            "sampling_method": "first",
        },
        "regime_consistency": {
            "enabled": True,
            "rul_matching_tolerance": 2.0,
            "maximum_regime_pairs": 4,
            "maximum_regime_anchors": 8,
            "maximum_partners_per_anchor": 1,
            "maximum_pairs_per_regime_combination": 3,
            "latent_distance": "cosine",
            "pair_seed": 9,
            "allow_empty_pairs": False,
            "lazy_build": True,
            "cache_bounded_pairs": True,
        },
        "losses": {"lambda_data": 1.0},
        "candidate_registry": {"maximum_candidate_count": 10, "definitions": None},
        "later_experiment": {
            "screening_split": {"validation_fraction": 0.5, "seed": 10},
            "validation_snapshot_positions": [0.5, 1.0],
            "finalist_count": 1,
            "cv_folds": 2,
            "seeds": [11],
            "maximum_epochs": 2,
            "optimizer": "adamw",
            "batch_size": 4,
            "num_workers": 0,
            "mixed_precision": "false",
            "gradient_clip_norm": 1.0,
            "training_schedules": {
                "schedule_b": {"learning_rate": 0.001, "weight_decay": 0.0, "max_epochs": 1, "minimum_epochs": 1, "early_stopping_patience": 0, "scheduler": "none"}
            },
            "robust_selection_weights": {"normalized_RMSE": 1.0},
        },
        "uncertainty": {"nominal_levels": [0.8, 0.9, 0.95], "predicted_rul_bands": [{"label": "all", "lower": 0, "upper": None}], "coverage_tolerance": 0.1},
        "safety": {
            "low_rul_threshold": 10,
            "severe_optimistic_threshold": 5,
            "support_settings": {"support_percentile_range": [0.01, 0.99], "limited_feature_exceedance": 0.1, "out_feature_exceedance": 0.5},
            "abstention_settings": {"max_feature_exceedance_fraction": 0.5, "max_interval_width_ratio": 5, "min_plausible_rul": 0, "max_plausible_rul": 100},
            "maintenance_thresholds": {"urgent_review_max": 5, "schedule_maintenance_max": 10, "plan_inspection_max": 20},
        },
        "smoke_test": {"synthetic_engine_count": 4, "synthetic_regime_count": 2, "synthetic_cycles_per_engine": 8, "synthetic_feature_count": 5, "smoke_epochs": 1, "smoke_batch_size": 4, "learning_rate": 0.001, "smoke_output_directory": str(tmp_path / "smoke")},
    }


def _prepare_phase5b_files(config: dict) -> None:
    phase5b_dir = Path(config["general"]["phase5b_results_path"])
    phase5b_dir.mkdir(parents=True)
    (phase5b_dir / "run_summary.json").write_text('{"locked_model_id":"patch","benchmark_metrics":{},"deep_uncertainty_metrics":{}}', encoding="utf-8")
    Path(config["general"]["phase5b_checkpoint_path"]).write_bytes(b"checkpoint")


def _screening_row(candidate_id: str, *, rmse: float = 2.0, nasa: float = 5.0, status: str = "success", reason: str = "") -> dict:
    return {
        "candidate_id": candidate_id,
        "architecture": "{}",
        "training_status": status,
        "failure_reason": reason,
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
        "validation_mae": 1.0,
        "validation_rmse": rmse,
        "validation_nasa_score": nasa,
        "validation_mean_signed_error": 0.0,
        "validation_optimistic_rate": 0.1,
        "validation_severe_optimistic_rate": 0.0,
        "validation_low_rul_optimistic_rate": 0.0,
        "monotonic_violation_rate": 0.0,
        "rate_violation_rate": 0.0,
        "smoothness_violation_rate": 0.0,
        "health_violation_rate": 0.0,
        "regime_consistency_violation_rate": 0.0,
        "parameter_count": 1000,
        "checkpoint_size": 100,
        "training_runtime": 0.1,
        "cpu_latency": 0.1,
        "gpu_latency": np.nan,
        "checkpoint_path": "",
    }


def _benchmark_frame(labels: tuple[float, float] = (7.0, 12.0)) -> pd.DataFrame:
    rows = []
    for unit, length, final_rul in [(1, 3, labels[0]), (2, 5, labels[1])]:
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
                    "true_rul_uncapped": float(final_rul + length - cycle),
                }
            )
    return pd.DataFrame(rows)


def _write_resume_artifacts(config: dict, tmp_path: Path, *, feature_names: list[str] | None = None, checkpoint_exists: bool = True) -> None:
    output_dir = Path(config["general"]["output_dir"])
    checkpoint_dir = Path(config["general"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    features = feature_names or ["sensor_1"]
    preprocessor_path = checkpoint_dir / "final_preprocessor.pkl"
    transformed_path = checkpoint_dir / "final_train_transformed.pkl"
    checkpoint_path = checkpoint_dir / "locked_physics_guided_model.pt"
    with preprocessor_path.open("wb") as handle:
        pickle.dump({"features": features}, handle)
    with transformed_path.open("wb") as handle:
        pickle.dump(pd.DataFrame({"sensor_1": [0.0]}), handle)
    if checkpoint_exists:
        torch.save({"state_dict": {}, "metadata": {"feature_names": features}}, checkpoint_path)
    pipeline.atomic_write_json(output_dir / "run_summary.json", {"run_status": "failed", "failed_stage": "evaluate_benchmark_subsets", "stage": "evaluate_benchmark_subsets"})
    pipeline.atomic_write_json(output_dir / "locked_physics_model.json", {"candidate_id": "physics_regime"})
    pipeline.atomic_write_json(
        output_dir / "final_fit_metadata.json",
        {
            "candidate_id": "physics_regime",
            "checkpoint_path": str(checkpoint_path),
            "preprocessor_path": str(preprocessor_path),
            "final_train_transformed_path": str(transformed_path),
            "feature_names": features,
            "operating_regime_metadata": {"method": "regime_standardization"},
            "rul_cap": float(config["sequence"]["rul_cap"]),
            "config_hash": pipeline.stable_payload_hash(config),
            "candidate_registry_hash": pipeline.stable_payload_hash(pipeline._candidate_registry(config)),
        },
    )
    pd.DataFrame([{"candidate_id": "physics_regime", "fold": 1, "seed": 11, "training_status": "success"}]).to_csv(output_dir / "finalist_cross_validation_metrics.csv", index=False)
    pd.DataFrame([{"candidate_id": "physics_regime", "true_rul": 1.0, "predicted_rul": 1.0, "residual": 0.0}]).to_csv(output_dir / "cv_predictions.csv", index=False)


def _mock_stages(order: list[str]):
    stages = {}
    for name in pipeline.FULL_RUN_STAGE_ORDER:
        def stage(state, name=name):
            order.append(name)
            if name == "write_run_summary":
                state["run_summary"] = {"run_status": "complete", "stage_order": list(order)}
                pipeline.atomic_write_json(state["output_dir"] / "run_summary.json", state["run_summary"])
                return state["run_summary"]
            return {"stage": name}
        stages[name] = stage
    return stages


def test_full_run_dispatches_to_full_experiment(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    called = {}
    monkeypatch.setattr(pipeline, "load_config", lambda path: config)
    def fake_full_experiment(cfg, config_path=None, root=None, resume_from=None):
        called["config"] = cfg
        return {"ok": True}
    monkeypatch.setattr(pipeline, "run_full_experiment", fake_full_experiment)

    result = pipeline.run_full_run(tmp_path / "config.yaml")

    assert result == {"ok": True}
    assert called["config"] is config


def test_mocked_full_experiment_executes_stage_order(tmp_path: Path) -> None:
    config = _config(tmp_path)
    order: list[str] = []

    result = pipeline.run_full_experiment(config, config_path=tmp_path / "config.yaml", root=tmp_path, stage_overrides=_mock_stages(order))

    assert result["run_status"] == "complete"
    assert order == pipeline.FULL_RUN_STAGE_ORDER


def test_failure_records_failure_summary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    stages = _mock_stages([])
    def fail(state):
        raise RuntimeError("boom")
    stages["screen_all_candidates"] = fail

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run_full_experiment(config, root=tmp_path, stage_overrides=stages)

    summary = Path(config["general"]["output_dir"]) / "run_summary.json"
    assert summary.exists()
    assert "failed" in summary.read_text(encoding="utf-8")


def test_benchmark_and_final_fit_stage_guards(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="Benchmark evaluation"):
        pipeline.evaluate_benchmark_subsets({"config": config})
    with pytest.raises(RuntimeError, match="Final fit"):
        pipeline.fit_final_physics_model({"config": config, "training_frame": pd.DataFrame()})


def test_candidate_selection_uses_validation_metrics_only(tmp_path: Path) -> None:
    config = _config(tmp_path)
    state = {
        "config": config,
        "output_dir": tmp_path,
        "screening_metrics": pd.DataFrame(
            [
                {"candidate_id": "a", "training_status": "success", "validation_rmse": 1.0, "benchmark_rmse": 99.0},
                {"candidate_id": "b", "training_status": "success", "validation_rmse": 2.0, "benchmark_rmse": 0.1},
            ]
        ),
        "generated_files": [],
    }

    finalists = pipeline.select_finalists(state)

    assert finalists.iloc[0]["candidate_id"] == "a"


def test_test_metric_weight_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["later_experiment"]["robust_selection_weights"] = {"benchmark_rmse": 1.0}
    frame = pd.DataFrame([{"candidate_id": "a", "training_status": "success", "validation_rmse": 1.0}])

    with pytest.raises(ValueError, match="Benchmark/test"):
        pipeline.rank_candidates_dataframe(frame, config, require_success=True)


def test_validation_metrics_calculate_canonical_nasa_score(tmp_path: Path) -> None:
    config = _config(tmp_path)
    frame = pd.DataFrame({"true_rul": [10.0, 20.0, 30.0], "predicted_rul": [9.0, 25.0, 29.0]})

    metrics = pipeline.validation_metrics_for_frame(frame, config)

    assert "validation_nasa_score" in metrics
    assert metrics["validation_nasa_score"] == pytest.approx(nasa_asymmetric_score(frame["true_rul"], frame["predicted_rul"]))
    assert metrics["validation_mean_signed_error"] == pytest.approx(1.0)


def test_canonical_screening_schema_and_failure_row(tmp_path: Path) -> None:
    config = _config(tmp_path)
    candidate = pipeline._candidate_registry(config)[0]
    state = {"screening_train_metadata": pd.DataFrame({"global_engine_id": ["a", "b"]}), "screening_validation_metadata": pd.DataFrame({"global_engine_id": ["c"]})}

    row = pipeline._failed_candidate_row(candidate, state, RuntimeError("boom"))
    normalized = pipeline.normalize_screening_metrics_schema(pd.DataFrame([row]))

    assert list(normalized.columns[: len(pipeline.CANONICAL_SCREENING_SCHEMA)]) == pipeline.CANONICAL_SCREENING_SCHEMA
    assert normalized.loc[0, "training_status"] == "failed"
    assert "boom" in normalized.loc[0, "failure_reason"]
    assert "validation_nasa_score" in normalized.columns


def test_schema_normalization_alias_conflict_and_no_mutation() -> None:
    original = pd.DataFrame([{"candidate_id": "a", "training_status": "success", "nasa_score": 3.0, "val_rmse": 2.0, "val_mae": 1.0}])
    snapshot = original.copy(deep=True)

    normalized = pipeline.normalize_screening_metrics_schema(original)

    pd.testing.assert_frame_equal(original, snapshot)
    assert normalized.loc[0, "validation_nasa_score"] == pytest.approx(3.0)
    assert normalized.loc[0, "validation_rmse"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="Conflicting metric alias"):
        pipeline.normalize_screening_metrics_schema(
            pd.DataFrame([{"candidate_id": "a", "training_status": "success", "validation_nasa_score": 4.0, "nasa_score": 3.0}])
        )


def test_ranking_diagnostics_for_missing_or_nonfinite_required_metric(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["later_experiment"]["robust_selection_weights"] = {"validation_nasa_score": 1.0}
    frame = pd.DataFrame([{"candidate_id": "a", "training_status": "success", "validation_rmse": 1.0, "failure_reason": ""}])

    with pytest.raises(ValueError, match="non_finite_required_metric:validation_nasa_score"):
        pipeline.rank_candidates_dataframe(frame, config, require_success=True)

    nonfinite = pd.DataFrame([_screening_row("a", nasa=np.nan)])
    with pytest.raises(ValueError, match="non_finite_required_metric:validation_nasa_score"):
        pipeline.rank_candidates_dataframe(nonfinite, config, require_success=True)

    failed = pd.DataFrame([_screening_row("a", status="failed", reason="training exploded")])
    with pytest.raises(ValueError, match="training exploded"):
        pipeline.rank_candidates_dataframe(failed, config, require_success=True)


def test_candidate_ranking_is_deterministic_and_nasa_lower_is_better(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["later_experiment"]["robust_selection_weights"] = {"validation_nasa_score": 1.0}
    frame = pd.DataFrame([_screening_row("b", rmse=1.0, nasa=10.0), _screening_row("a", rmse=2.0, nasa=5.0)])

    first = pipeline.rank_candidates_dataframe(frame, config, require_success=True)
    second = pipeline.rank_candidates_dataframe(frame.iloc[::-1].reset_index(drop=True), config, require_success=True)

    assert first["candidate_id"].tolist() == ["a", "b"]
    assert second["candidate_id"].tolist() == ["a", "b"]
    assert np.isfinite(first.filter(like="ranking_contribution_").to_numpy(dtype=float)).all()


def test_cv_schema_and_stability_use_canonical_nasa(tmp_path: Path) -> None:
    config = _config(tmp_path)
    cv = pd.DataFrame(
        [
            {**_screening_row("a", nasa=5.0), "fold": 1, "seed": 11},
            {**_screening_row("a", nasa=7.0), "fold": 2, "seed": 11},
        ]
    )
    normalized = pipeline.normalize_cv_metrics_schema(cv)
    state = {"cv_metrics": normalized, "output_dir": tmp_path, "generated_files": []}

    stability = pipeline.aggregate_stability_results(state)

    assert "validation_nasa_score" in normalized.columns
    assert "mean_validation_nasa_score" in stability.columns
    assert stability.loc[0, "mean_validation_nasa_score"] == pytest.approx(6.0)


def test_benchmark_endpoint_labels_align_and_reject_mismatch() -> None:
    frame = _benchmark_frame()
    endpoints = pipeline.build_benchmark_endpoint_table(frame, rul_cap=20.0, rul_values=[7.0, 12.0])

    assert endpoints["global_engine_id"].tolist() == ["FD001_0001", "FD001_0002"]
    assert endpoints["final_observed_cycle"].tolist() == [3, 5]
    assert endpoints["true_rul_capped"].tolist() == [7.0, 12.0]
    with pytest.raises(ValueError, match="RUL-file row count mismatch"):
        pipeline.build_benchmark_endpoint_table(frame, rul_cap=20.0, rul_values=[1.0])
    duplicated = pd.concat([frame, frame[frame["global_engine_id"] == "FD001_0001"].tail(1)], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        pipeline.build_benchmark_endpoint_table(duplicated, rul_cap=20.0, rul_values=[7.0, 12.0])


def test_benchmark_labels_attach_after_inference_and_do_not_change_tensors_or_predictions(tmp_path: Path) -> None:
    frame_a = _benchmark_frame((7.0, 12.0))
    frame_b = _benchmark_frame((17.0, 22.0))
    endpoints_a = pipeline.build_benchmark_endpoint_table(frame_a, rul_cap=125.0)
    endpoints_b = pipeline.build_benchmark_endpoint_table(frame_b, rul_cap=125.0)
    sensor_a = pipeline.benchmark_sensor_frame_without_labels(frame_a)
    sensor_b = pipeline.benchmark_sensor_frame_without_labels(frame_b)
    spec = pipeline.WindowSpec(window_length=4, stride=1, minimum_valid_history=1)

    dataset_a, metadata_a, sequences_a = pipeline.make_dataset(sensor_a, endpoints_a[["global_engine_id", "endpoint_index"]], ["sensor_1", "sensor_2"], spec, mode="inference")
    dataset_b, metadata_b, sequences_b = pipeline.make_dataset(sensor_b, endpoints_b[["global_engine_id", "endpoint_index"]], ["sensor_1", "sensor_2"], spec, mode="inference")
    pred = pd.DataFrame({"predicted_rul_raw": [8.0, 11.0], "predicted_rul": [8.0, 11.0], "health_score": np.nan, "degradation_rate": np.nan})
    base_predictions = pd.concat([metadata_a.reset_index(drop=True), pred], axis=1)
    base_predictions["subset"] = "FD001"
    base_predictions["final_observed_cycle"] = base_predictions["cycle"].astype(int)
    labeled_a = pipeline.attach_benchmark_labels(base_predictions, endpoints_a)
    labeled_b = pipeline.attach_benchmark_labels(base_predictions, endpoints_b)
    metrics_a = pipeline.deep_point_metrics(labeled_a["true_rul"], labeled_a["predicted_rul"], 30.0)
    metrics_b = pipeline.deep_point_metrics(labeled_b["true_rul"], labeled_b["predicted_rul"], 30.0)

    assert not any(column in sensor_a.columns for column in pipeline.BENCHMARK_LABEL_COLUMNS)
    assert "target_rul_capped" not in metadata_a.columns
    np.testing.assert_allclose(sequences_a, sequences_b)
    np.testing.assert_allclose(dataset_a.sequences.numpy(), dataset_b.sequences.numpy())
    assert labeled_a["predicted_rul"].tolist() == labeled_b["predicted_rul"].tolist()
    assert metrics_a["mae"] != metrics_b["mae"]


def test_cv_prediction_concat_filters_empty_without_futurewarning() -> None:
    non_empty = pd.DataFrame({"candidate_id": ["a"], "true_rul": [1.0], "predicted_rul": [1.0], "residual": [0.0], "all_na_optional": [np.nan]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pipeline.normalize_cv_prediction_frames([pd.DataFrame(), non_empty])

    assert len(result) == 1
    assert "all_na_optional" in result.columns
    assert not any("DataFrame concatenation" in str(item.message) for item in caught)


def test_resume_inspection_accepts_complete_mocked_state_and_rejects_missing_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_resume_artifacts(config, tmp_path)

    report = pipeline.inspect_phase5c_resume_state(config, tmp_path)

    assert report["safe_to_resume"] is True
    assert report["earliest_safe_resume_stage"] == "evaluate_benchmark_subsets"

    config_missing = _config(tmp_path / "missing")
    _write_resume_artifacts(config_missing, tmp_path / "missing", checkpoint_exists=False)
    missing = pipeline.inspect_phase5c_resume_state(config_missing, tmp_path / "missing")
    assert missing["safe_to_resume"] is False
    assert "final_checkpoint" in missing["missing_or_invalid_artifacts"]


def test_resume_inspection_rejects_config_hash_and_feature_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_resume_artifacts(config, tmp_path)
    meta_path = Path(config["general"]["output_dir"]) / "final_fit_metadata.json"
    payload = pipeline._read_json(meta_path)
    payload["config_hash"] = "wrong"
    pipeline.atomic_write_json(meta_path, payload)

    report = pipeline.inspect_phase5c_resume_state(config, tmp_path)
    assert "config_hash_mismatch" in report["missing_or_invalid_artifacts"]

    config_features = _config(tmp_path / "features")
    _write_resume_artifacts(config_features, tmp_path / "features")
    feature_meta = pipeline._read_json(Path(config_features["general"]["output_dir"]) / "final_fit_metadata.json")
    feature_meta["feature_names"] = ["sensor_2"]
    pipeline.atomic_write_json(Path(config_features["general"]["output_dir"]) / "final_fit_metadata.json", feature_meta)
    feature_report = pipeline.inspect_phase5c_resume_state(config_features, tmp_path / "features")
    assert "feature_schema_mismatch" in feature_report["missing_or_invalid_artifacts"]


def test_select_finalists_accepts_legacy_nasa_alias_and_writes_diagnostics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["later_experiment"]["robust_selection_weights"] = {"validation_nasa_score": 1.0}
    row = _screening_row("a", nasa=5.0)
    row["nasa_score"] = row.pop("validation_nasa_score")
    state = {"config": config, "screening_metrics": pd.DataFrame([row]), "output_dir": tmp_path, "generated_files": []}

    finalists = pipeline.select_finalists(state)

    assert finalists["candidate_id"].tolist() == ["a"]
    assert (tmp_path / "finalist_selection_diagnostics.json").exists()


def test_mocked_full_run_advances_beyond_select_finalists_with_nasa(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["later_experiment"]["robust_selection_weights"] = {"validation_nasa_score": 1.0}
    order: list[str] = []
    stages = _mock_stages(order)

    def screen_success(state):
        order.append("screen_all_candidates")
        frame = pipeline.normalize_screening_metrics_schema(pd.DataFrame([_screening_row("a", nasa=5.0), _screening_row("b", nasa=6.0)]))
        state["screening_metrics"] = frame
        return frame

    stages["screen_all_candidates"] = screen_success
    stages.pop("select_finalists")

    result = pipeline.run_full_experiment(config, root=tmp_path, stage_overrides=stages)

    assert result["run_status"] == "complete"
    assert "run_finalist_cross_validation" in order
    assert (Path(config["general"]["output_dir"]) / "finalist_selection.json").exists()


def test_output_directory_lifecycle_and_completed_protection(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = Path(config["general"]["output_dir"])

    summary = pipeline.dry_run_orchestration_summary(config, tmp_path / "config.yaml", tmp_path)
    assert summary["dry_run_created_output_dir"] is False
    assert not output_dir.exists()

    pipeline.prepare_full_run_outputs(config, tmp_path)
    assert output_dir.exists()
    pipeline.atomic_write_json(output_dir / "run_summary.json", {"run_status": "complete"})
    with pytest.raises(FileExistsError):
        pipeline.prepare_full_run_outputs(config, tmp_path)


def test_partial_output_directory_requires_resume_or_overwrite(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = Path(config["general"]["output_dir"])
    output_dir.mkdir(parents=True)
    (output_dir / "pairing_audit.csv").write_text("partial", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Partial Phase 5C output"):
        pipeline.prepare_full_run_outputs(config, tmp_path)


def test_phase5b_hash_mismatch_stops_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _prepare_phase5b_files(config)
    state = {"config": config, "root": tmp_path, "phase5b_initial_hashes": {str(Path(config["general"]["phase5b_checkpoint_path"])): "wrong"}}

    with pytest.raises(RuntimeError, match="hash changed"):
        pipeline.verify_phase5b_hashes_unchanged(state)


def test_benchmark_rows_rejected_for_preprocessing() -> None:
    with pytest.raises(ValueError, match="Benchmark/test"):
        pipeline.assert_training_only_preprocessing_frame(pd.DataFrame({"data_role": ["benchmark_test"], "subset": ["FD001"]}))


def test_required_future_outputs_registered() -> None:
    assert "run_summary.json" in pipeline.FULL_RUN_OUTPUT_FILES
    assert "phase5b_benchmark_manifest.json" in pipeline.FULL_RUN_OUTPUT_FILES
    assert "locked_physics_guided_model.pt" in pipeline.FULL_RUN_CHECKPOINT_FILES


def test_regime_candidates_only_and_cache_scoped_by_split_seed(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    metadata = pd.DataFrame(
        {
            "sample_index": [0, 1, 2],
            "target_rul_capped": [10.0, 10.5, 11.0],
            "operating_regime": [0, 1, 2],
            "sequence_valid_length": [5, 5, 5],
            "global_engine_id": ["a", "b", "c"],
            "subset": ["FD001", "FD002", "FD003"],
        }
    )
    calls: list[str] = []

    def fake_build_regime_pairs(frame: pd.DataFrame, regime_config: pipeline.RegimePairingConfig) -> pd.DataFrame:
        calls.append(f"{len(frame)}:{regime_config.seed}:{regime_config.max_pairs}")
        pairs = pd.DataFrame({"left_index": [0], "right_index": [1]})
        pairs.attrs["diagnostics"] = {
            "metadata_rows": len(frame),
            "number_of_regimes": 3,
            "anchor_count_considered": 2,
            "pair_table_memory_mb": 0.001,
            "limit_reached": False,
        }
        return pairs

    monkeypatch.setattr(pipeline, "build_regime_pairs", fake_build_regime_pairs)
    registry = {candidate["candidate_id"]: candidate for candidate in pipeline._candidate_registry(config)}
    state: dict = {"config": config, "regime_pair_cache": {}, "warnings": []}

    assert pipeline.candidate_requires_regime_pairs(registry["physics_regime"]) is True
    assert pipeline.candidate_requires_regime_pairs(registry["physics_full"]) is True
    assert pipeline.candidate_requires_regime_pairs(registry["physics_full_safety"]) is True
    assert pipeline.candidate_requires_regime_pairs(registry["phase5b_reimplementation_baseline"]) is False
    assert pipeline.candidate_requires_regime_pairs(registry["physics_temporal_combined"]) is False

    baseline_pairs = pipeline.get_regime_pairs_for_candidate(state, registry["phase5b_reimplementation_baseline"], "screening", metadata)
    first = pipeline.get_regime_pairs_for_candidate(state, registry["physics_regime"], "cv_fold0_seed11", metadata)
    second = pipeline.get_regime_pairs_for_candidate(state, registry["physics_full"], "cv_fold0_seed11", metadata)
    third = pipeline.get_regime_pairs_for_candidate(state, registry["physics_full"], "cv_fold0_seed12", metadata)

    assert baseline_pairs.empty
    assert first is second
    assert len(third) == 1
    assert state["regime_pair_cache_hits"] == 1
    assert len(calls) == 2


def test_fully_mocked_full_run_creates_temporary_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    order: list[str] = []

    result = pipeline.run_full_experiment(config, root=tmp_path, stage_overrides=_mock_stages(order))

    assert result["run_status"] == "complete"
    assert (Path(config["general"]["output_dir"]) / "run_summary.json").exists()
    assert Path(config["general"]["output_dir"]).is_relative_to(tmp_path)
