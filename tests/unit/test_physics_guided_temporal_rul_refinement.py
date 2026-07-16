from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from aeroguard.deep.models.patch_transformer import PatchTemporalTransformerRegressor
from aeroguard.evaluation.deep_rul_metrics import metrics_by_group
from aeroguard.pipelines.refine_physics_guided_temporal_rul import (
    abstention_policy_selection,
    apply_conformal_policy,
    apply_locked_maintenance,
    bootstrap_constraint_intervals,
    build_regime_comparison_details,
    build_temporal_comparison_details,
    corrected_constraint_diagnostics,
    maintenance_policy_selection,
    paired_bootstrap_comparison,
    risk_feature_frame,
    run_smoke_test,
    true_condition_band,
    uncertainty_candidate_selection,
)
from aeroguard.pipelines.train_physics_guided_temporal_rul import (
    load_config as load_phase5c_config,
    normalize_cv_prediction_frames,
    run_full_experiment,
)


def _refinement_config() -> dict:
    return {
        "constraint_audit": {
            "rul_matching_tolerance": 5.0,
            "maximum_regime_pairs": 100,
            "monotonic_tolerance": 0.5,
            "rate_threshold_quantile": 0.90,
            "smoothness_threshold_quantile": 0.90,
            "regime_threshold_quantile": 0.90,
            "bootstrap_iterations": 20,
            "bootstrap_seed": 11,
        },
        "policy_selection": {"development_fraction": 0.7, "split_seeds": [1, 2, 3], "group_by_engine": True},
        "uncertainty": {"nominal_levels": [0.8, 0.9, 0.95], "minimum_group_size": 3, "maximum_undercoverage": 0.05, "maximum_instability": 1.0, "shrinkage_weight": 0.5},
        "abstention": {"high_error_threshold": 25.0, "maximum_abstention_rate": 0.15, "minimum_error_enrichment": 1.1, "allow_no_abstention": True},
        "maintenance": {
            "critical_rul_max": 15,
            "near_term_rul_max": 30,
            "inspection_rul_max": 60,
            "healthy_rul_min": 125,
            "critical_recall_floor": 0.8,
            "cost_matrix": {"missed_critical": 100.0, "delayed_near_term": 35.0, "unnecessary_urgent": 8.0, "early_inspection": 2.0, "wrong_noncritical_action": 4.0},
        },
        "bootstrap": {"paired_iterations": 20, "seed": 21},
    }


def _validation_frame() -> pd.DataFrame:
    rows = []
    for subset in ["FD001", "FD002"]:
        for engine in range(1, 9):
            regime = engine % 3
            for idx, cycle in enumerate([10, 20, 40, 70]):
                true = float(120 - cycle + engine)
                pred = true + (engine % 4 - 1.5) * 4.0 + idx * 0.5
                rows.append(
                    {
                        "subset": subset,
                        "source_domain": subset,
                        "global_engine_id": f"{subset}_{engine:04d}",
                        "local_unit_id": engine,
                        "unit_id": engine,
                        "cycle": cycle,
                        "endpoint_index": idx,
                        "endpoint_cycle": cycle,
                        "sequence_valid_length": min(cycle, 50),
                        "padded_cycle_count": max(0, 50 - cycle),
                        "target_rul_capped": min(100.0, true),
                        "target_rul_uncapped": true,
                        "operating_regime": regime,
                        "predicted_rul_raw": pred,
                        "predicted_rul": pred,
                        "candidate_id": "physics_regime",
                        "true_rul": true,
                        "residual": pred - true,
                        "absolute_error": abs(pred - true),
                        "squared_error": (pred - true) ** 2,
                        "prediction_direction": "over" if pred > true else "under",
                        "fold": 1 + engine % 2,
                        "seed": 100 + engine % 2,
                        "sample_index": len(rows),
                        "final_observed_cycle": cycle,
                    }
                )
    return pd.DataFrame(rows)


def test_successful_future_run_summary_becomes_completed(tmp_path: Path) -> None:
    config = load_phase5c_config("configs/physics_guided_temporal_rul.yaml")
    config["general"]["output_dir"] = str(tmp_path / "reports")
    config["general"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    config["general"]["overwrite_existing"] = True

    def noop(state: dict) -> dict:
        return {"ok": True}

    overrides = {name: noop for name in [
        "inspect_environment",
        "verify_phase5b_artifacts",
        "create_phase5b_manifest",
        "load_training_subsets_stage",
        "load_benchmark_subsets_stage",
        "create_screening_split_stage",
        "fit_fold_preprocessing_stage",
        "create_standard_windows_stage",
        "create_temporal_pairs_stage",
        "create_regime_pairs_stage",
        "screen_all_candidates",
        "select_finalists",
        "run_finalist_cross_validation",
        "aggregate_stability_results",
        "run_constraint_ablation_analysis",
        "rank_physics_candidates",
        "lock_physics_model",
        "determine_locked_epoch_count",
        "fit_final_physics_model",
        "evaluate_benchmark_subsets",
        "evaluate_trajectory_consistency",
        "run_optimistic_error_analysis",
        "compare_phase5b_predictions",
        "fit_deep_conformal_calibration",
        "evaluate_uncertainty",
        "evaluate_support_and_abstention",
        "generate_maintenance_recommendations",
        "measure_model_efficiency",
        "generate_figures",
        "write_results_documentation",
        "verify_phase5b_hashes_unchanged",
    ]}
    result = run_full_experiment(config, root=Path.cwd(), stage_overrides=overrides)
    written = json.loads((tmp_path / "reports" / "run_summary.json").read_text(encoding="utf-8"))
    assert result["run_status"] == "completed"
    assert written["runtime_seconds"] > 0
    assert written["completed_stage_count"] == 32
    assert written["failed_stage"] is None


def test_failed_future_run_summary_becomes_failed(tmp_path: Path) -> None:
    config = load_phase5c_config("configs/physics_guided_temporal_rul.yaml")
    config["general"]["output_dir"] = str(tmp_path / "reports")
    config["general"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    config["general"]["overwrite_existing"] = True

    def fail(_: dict) -> None:
        raise RuntimeError("planned failure")

    try:
        run_full_experiment(config, root=Path.cwd(), stage_overrides={"inspect_environment": fail})
    except RuntimeError:
        pass
    written = json.loads((tmp_path / "reports" / "run_summary.json").read_text(encoding="utf-8"))
    assert written["run_status"] == "failed"
    assert written["failed_stage"] == "inspect_environment"
    assert written["exception_type"] == "RuntimeError"
    assert written["runtime_seconds"] > 0


def test_concat_and_groupby_warning_paths_are_clean() -> None:
    frame = pd.DataFrame({"subset": pd.Categorical(["a", "b"]), "true_rul": [1.0, 2.0], "predicted_rul": [1.5, 1.5]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics_by_group(frame, "subset")
        normalize_cv_prediction_frames([pd.DataFrame(), pd.DataFrame({"subset": ["x"], "predicted_rul": [1.0], "true_rul": [1.0]})])
    assert not any("FutureWarning" in type(item.message).__name__ or "observed" in str(item.message) for item in caught)


def test_transformer_warning_removed_and_predictions_stable() -> None:
    model = PatchTemporalTransformerRegressor(input_dim=4, window_length=6, patch_length=3, patch_stride=3, projection_dim=8, layers=1, heads=2, feedforward_dim=16, dropout=0.0)
    clone = PatchTemporalTransformerRegressor(input_dim=4, window_length=6, patch_length=3, patch_stride=3, projection_dim=8, layers=1, heads=2, feedforward_dim=16, dropout=0.0)
    clone.load_state_dict(model.state_dict())
    model.eval()
    clone.eval()
    x = torch.randn(2, 6, 4)
    x[..., -1] = 1.0
    with warnings.catch_warnings(record=True) as caught, torch.no_grad():
        warnings.simplefilter("always")
        first = model(x)
        second = clone(x)
    assert torch.allclose(first, second, rtol=1e-6, atol=1e-6)
    assert not any("nested" in str(item.message).lower() for item in caught)


def test_corrected_constraint_formulas_and_regime_pairs() -> None:
    frame = _validation_frame()
    pairs, triplets = build_temporal_comparison_details(frame, cap=100.0, mono_tolerance=0.5)
    assert (pairs["later_cycle"] > pairs["earlier_cycle"]).all()
    assert (pairs["monotonic_violation_magnitude"] >= 0).all()
    plateau = pairs[pairs["trajectory_region"] == "healthy_capped_plateau"]
    assert not plateau.empty
    assert (plateau["expected_capped_delta"] == 0.0).all()
    assert not triplets.empty
    assert (triplets["abs_acceleration"] >= 0).all()
    regime = build_regime_comparison_details(frame, tolerance=5.0, max_pairs=100, seed=4)
    assert not regime.empty
    assert (regime["left_regime"] != regime["right_regime"]).all()
    assert (regime["true_rul_difference"].abs() <= 5.0).all()
    one_regime = frame.assign(operating_regime=1)
    assert build_regime_comparison_details(one_regime, tolerance=5.0, max_pairs=100, seed=4).empty


def test_constraint_thresholds_validation_and_engine_bootstrap() -> None:
    audit, metrics, details, thresholds = corrected_constraint_diagnostics(_validation_frame(), _refinement_config())
    assert thresholds["cycle_rate"] > 0
    assert thresholds["smoothness"] >= 0
    assert audit["cycle_rate"]["corrected_threshold_selection_method"].startswith("validation-only")
    intervals = bootstrap_constraint_intervals(details, thresholds, iterations=10, seed=5)
    assert set(intervals["metric_name"]) == {"monotonicity", "cycle_rate", "smoothness", "regime_consistency"}
    assert (intervals["engine_count"] > 0).all()


def test_uncertainty_selection_is_deterministic_and_label_locked() -> None:
    frame = _validation_frame()
    metrics_a, locked_a, cv_unc_a = uncertainty_candidate_selection(frame, _refinement_config())
    metrics_b, locked_b, _ = uncertainty_candidate_selection(frame, _refinement_config())
    assert locked_a["candidate_method"] == locked_b["candidate_method"]
    benchmark = frame.groupby(["subset", "global_engine_id"], observed=False).tail(1).copy()
    pred_a = apply_conformal_policy(benchmark, locked_a)
    altered = benchmark.copy()
    altered["true_rul"] += 50.0
    pred_b = apply_conformal_policy(altered, locked_a)
    assert np.allclose(pred_a["lower_90"], pred_b["lower_90"])
    assert np.allclose(pred_a["upper_90"], pred_b["upper_90"])
    assert "coverage" in set(metrics_a.columns)
    assert "lower_90" in cv_unc_a.columns


def test_abstention_no_abstention_can_win_and_enrichment_calculates(tmp_path: Path) -> None:
    _, locked_uncertainty, cv_unc = uncertainty_candidate_selection(_validation_frame(), _refinement_config())
    candidates, locked, _ = abstention_policy_selection(cv_unc, _refinement_config(), tmp_path)
    assert "no_abstention" in set(candidates["policy_id"].dropna())
    assert locked["policy_id"] == "no_abstention" or float(candidates["error_enrichment_ratio"].max()) >= 1.0


def test_maintenance_policy_uses_prediction_inputs_and_enforces_floor() -> None:
    config = _refinement_config()
    _, locked_uncertainty, cv_unc = uncertainty_candidate_selection(_validation_frame(), config)
    candidates, locked = maintenance_policy_selection(cv_unc, config)
    assert locked["selection_source"].startswith("engine-grouped")
    assert "critical_recall" in candidates.columns
    benchmark = cv_unc.groupby(["subset", "global_engine_id"], observed=False).tail(1).copy()
    altered = benchmark.copy()
    altered["true_rul"] += 1000.0
    actions_a = apply_locked_maintenance(risk_feature_frame(benchmark), locked, config)["maintenance_action"]
    actions_b = apply_locked_maintenance(risk_feature_frame(altered), locked, config)["maintenance_action"]
    assert actions_a.equals(actions_b)
    assert set(true_condition_band(pd.Series([5, 20, 45, 80, 150]), config)) == {"critical", "near_term", "inspection", "monitoring", "healthy"}


def test_paired_bootstrap_aligns_engines_and_rejects_duplicates() -> None:
    frame = _validation_frame().groupby(["subset", "global_engine_id"], observed=False).tail(1).copy().reset_index(drop=True)
    p5b = frame.copy()
    p5b["predicted_rul"] += 2.0
    p5b["residual"] = p5b["predicted_rul"] - p5b["true_rul"]
    p5b["absolute_error"] = p5b["residual"].abs()
    p5b["squared_error"] = np.square(p5b["residual"])
    frame["lower_90"] = np.maximum(0.0, frame["predicted_rul"] - 10)
    frame["upper_90"] = frame["predicted_rul"] + 10
    frame["interval_width_90"] = frame["upper_90"] - frame["lower_90"]
    frame["covered_90"] = (frame["true_rul"] >= frame["lower_90"]) & (frame["true_rul"] <= frame["upper_90"])
    p5b_unc = p5b.copy()
    p5b_unc["interval_width_90"] = 30.0
    p5b_unc["covered_90"] = True
    first = paired_bootstrap_comparison(p5b, p5b_unc, frame, iterations=10, seed=1)
    second = paired_bootstrap_comparison(p5b, p5b_unc, frame, iterations=10, seed=1)
    assert first["absolute_difference"].equals(second["absolute_difference"])
    duplicate = pd.concat([p5b, p5b.iloc[[0]]], ignore_index=True)
    try:
        paired_bootstrap_comparison(duplicate, p5b_unc, frame, iterations=10, seed=1)
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("duplicate engine key was accepted")


def test_refinement_smoke_completes_without_neural_training(tmp_path: Path) -> None:
    config = {
        "source_phase5c": {"reports_dir": "unused", "artifacts_dir": "unused", "expected_locked_candidate": "physics_regime"},
        "source_phase5b": {"reports_dir": "unused", "artifacts_dir": "unused"},
        "outputs": {"reports_dir": str(tmp_path / "reports"), "artifacts_dir": str(tmp_path / "artifacts"), "overwrite_existing": True},
        **_refinement_config(),
    }
    config_path = tmp_path / "refine.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = run_smoke_test(config_path)
    assert result["status"] == "smoke_complete"
    assert result["neural_training_function_called"] is False
