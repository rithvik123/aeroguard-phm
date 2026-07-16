from pathlib import Path

from aeroguard.pipelines.train_physics_guided_temporal_rul import run_dry_run, run_smoke_test, run_validate_config


def test_physics_guided_temporal_rul_validate_dry_run_and_smoke() -> None:
    config_path = Path("configs/physics_guided_temporal_rul.yaml")

    validation = run_validate_config(config_path)
    dry_run = run_dry_run(config_path)
    smoke = run_smoke_test(config_path)

    assert validation["status"] == "valid"
    assert dry_run["status"] == "dry_run_complete"
    assert dry_run["parameter_count"] > 0
    assert dry_run["regime_pair_algorithm"] == "bounded_rul_searchsorted"
    assert dry_run["regime_pair_lazy_build"] is True
    assert dry_run["regime_pair_caps"]["maximum_regime_pairs"] == 20000
    assert dry_run["candidates_requiring_regime_pairs"] == ["physics_regime", "physics_full", "physics_full_safety"]
    assert dry_run["unbounded_regime_pair_generation_remaining"] is False
    assert "validation_nasa_score" in dry_run["canonical_screening_metric_names"]
    assert "validation_nasa_score" in dry_run["canonical_cv_metric_names"]
    assert dry_run["ranking_metrics_recognized"] is True
    assert dry_run["nasa_score_calculator_resolves"] is True
    assert dry_run["screening_serialization_includes_nasa_score"] is True
    assert dry_run["cv_serialization_includes_nasa_score"] is True
    assert smoke["status"] == "smoke_complete"
    assert smoke["synthetic_only"] is True
    assert smoke["gradient_seen"] is True
    assert smoke["padded_value_invariance"] is True
    assert smoke["reload_prediction_agreement"] is True
    assert smoke["no_future_cycle_leakage"] is True
    assert smoke["lazy_regime_non_regime_pair_count"] == 0
    assert smoke["lazy_regime_pair_count"] <= 20000
    assert smoke["lazy_regime_cache_hits"] >= 1
    assert smoke["lazy_regime_cache_reused"] is True
    assert smoke["lazy_regime_empty_reason"] == "only_one_regime"
    assert smoke["smoke_screening_candidate_count"] == 2
    assert smoke["smoke_validation_nasa_score_present"] is True
    assert smoke["smoke_validation_nasa_score_finite"] is True
    assert smoke["smoke_finalist_selection_complete"] is True
    assert smoke["smoke_ranking_contributions_finite"] is True
    assert smoke["smoke_benchmark_frame_has_rul_capped"] is False
    assert smoke["smoke_benchmark_inference_window_count"] == 2
    assert smoke["smoke_benchmark_inference_has_targets"] is False
    assert smoke["smoke_benchmark_label_free_tensor"] is True
    assert smoke["smoke_benchmark_predictions_label_invariant"] is True
    assert smoke["smoke_benchmark_metrics_change_with_labels"] is True
    assert smoke["temporary_files_limited_to_smoke_directory"] is True
    assert smoke["temporary_directory_removed"] is True
