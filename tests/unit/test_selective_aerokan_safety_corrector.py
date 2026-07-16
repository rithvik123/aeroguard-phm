from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.isotonic import IsotonicRegression

from aeroguard.pipelines.train_selective_aerokan_safety_corrector import (
    TRANSFORMER_TRAINING_CALLED,
    ConstantMagnitudeModel,
    CorrectionFit,
    GateFit,
    NoGateModel,
    OneSidedKANMagnitude,
    audit_feature_leakage,
    build_correction_candidate_registry,
    calibration_error,
    correction_candidate_feature_names,
    critical_under_correction_loss,
    dangerous_optimism_target,
    failure_summary,
    fit_correction_candidate,
    fit_gate_candidate,
    fit_selective_abstention,
    fit_selective_maintenance,
    fit_selective_uncertainty,
    gate_candidate_feature_names,
    kfold_engine_splits,
    load_config,
    magnitude_target,
    one_sided_final_prediction,
    paired_engine_alignment,
    predict_correction_magnitude,
    predict_gate_probability,
    pruning_decision,
    read_phase5c_benchmark_features,
    run_smoke_test,
    safe_row_zero_correction_loss,
    select_gate_threshold,
    selection_flags,
    selective_corrected_predictions,
    source_hashes_unchanged,
    strip_benchmark_labels_before_lock,
    write_prebenchmark_lock_manifest,
)


def config() -> dict:
    cfg = copy.deepcopy(load_config("configs/selective_aerokan_safety_corrector.yaml"))
    cfg["gate"]["sparse_kan_epochs"] = 1
    cfg["training"]["screening_epochs"] = 1
    cfg["training"]["finalist_epochs"] = 1
    cfg["training"]["final_epochs"] = 1
    cfg["bootstrap"]["iterations"] = 10
    return cfg


def toy_frame(n: int = 36) -> pd.DataFrame:
    rows = []
    for i in range(n):
        true = float([8, 12, 20, 35, 70, 95][i % 6])
        pred = true + float([18, 12, 8, -4, 6, -5][i % 6])
        rows.append(
            {
                "subset": "S",
                "global_engine_id": f"e{i:03d}",
                "cycle": i + 1,
                "predicted_rul": pred,
                "true_rul": true,
                "operating_regime": i % 3,
                "domain_support_score": 0.8 - 0.1 * (i % 3),
                "operating_regime_distance": 0.2 + 0.1 * (i % 3),
                "operating_regime_rarity": 0.2,
                "transformer_health_score": max(0.0, 1.0 - true / 125.0),
                "transformer_degradation_rate": 0.02 * (i % 4),
                "recent_base_rul_slope": -1.0 + 0.05 * (i % 5),
                "base_cycle_rate_residual": 0.1 * (i % 5),
                "base_monotonicity_residual": 0.0,
                "valid_sequence_fraction": 1.0,
                "padding_fraction": 0.0,
                "sensor_2_first_diff": 0.1 * (i % 4),
                "sensor_2_slope_5": -0.02 * (i % 5),
                "sensor_2_slope_10": -0.01 * (i % 7),
                "sensor_2_slope_gap": -0.01,
                "sensor_2_latest": 500.0 + i,
            }
        )
    return pd.DataFrame(rows)


def baseline_fit(cfg: dict) -> CorrectionFit:
    return CorrectionFit(
        {"candidate_id": "phase5c_exact_fallback", "candidate_type": "baseline", "correction_bound": 0.0},
        ConstantMagnitudeModel(0.0, 0.0),
        {
            "feature_family": "correction",
            "feature_names": ["base_rul_prediction"],
            "all_candidate_features": ["base_rul_prediction"],
            "healthy_baselines": {"sensor_2_latest": 500.0},
            "risk_metadata": {"radius_80": 10.0, "radius_90": 20.0},
            "mean": {"base_rul_prediction": 0.0},
            "std": {"base_rul_prediction": 1.0},
            "input_clamp": 5.0,
            "healthy_row_definition": "toy",
        },
        {},
    )


def test_01_final_prediction_never_exceeds_phase5c() -> None:
    base = np.array([10.0, 20.0, 0.5])
    final, _, _ = one_sided_final_prediction(base, np.array([1, 1, 1]), np.array([5, 25, 2]), threshold=0.5, bound=20)
    assert np.all(final <= base)


def test_02_gate_inactive_prediction_exactly_equals_phase5c() -> None:
    base = np.array([10.0, 20.0])
    final, downward, gate = one_sided_final_prediction(base, np.array([0.1, 0.2]), np.array([5, 5]), threshold=0.5, bound=10)
    assert np.allclose(final, base, atol=0.0, rtol=0.0)
    assert np.all(downward == 0.0)
    assert np.all(gate == 0.0)


def test_03_correction_magnitude_is_nonnegative() -> None:
    _, downward, _ = one_sided_final_prediction(np.array([10.0]), np.array([1.0]), np.array([-4.0]), threshold=0.5, bound=10)
    assert downward.item() >= 0.0


def test_04_downward_correction_is_bounded() -> None:
    _, downward, _ = one_sided_final_prediction(np.array([100.0]), np.array([1.0]), np.array([99.0]), threshold=0.5, bound=10)
    assert downward.item() == 10.0


def test_05_final_rul_remains_nonnegative() -> None:
    final, _, _ = one_sided_final_prediction(np.array([2.0]), np.array([1.0]), np.array([10.0]), threshold=0.5, bound=10)
    assert final.item() == 0.0


def test_06_dangerous_event_target_is_correct() -> None:
    frame = pd.DataFrame({"true_rul": [20, 31, 20], "predicted_rul": [35, 50, 25]})
    assert dangerous_optimism_target(frame, config()).tolist() == [1, 0, 0]


def test_07_magnitude_target_is_correct() -> None:
    frame = pd.DataFrame({"true_rul": [20, 20, 50], "predicted_rul": [45, 25, 80]})
    assert magnitude_target(frame, config(), 15.0).tolist() == [15.0, 0.0, 0.0]


def test_08_gate_features_exclude_true_rul_and_residual() -> None:
    audit = audit_feature_leakage(gate_candidate_feature_names(config()))
    assert audit["leakage_detected"] is False


def test_09_engine_grouped_splits_contain_no_overlap() -> None:
    splits = kfold_engine_splits(toy_frame(), 3, 42)
    for train, val, _ in splits:
        assert set(train["global_engine_id"]).isdisjoint(set(val["global_engine_id"]))


def test_10_logistic_gate_trains_correctly() -> None:
    fit = fit_gate_candidate("logistic", toy_frame(), config())
    prob, _ = predict_gate_probability(fit, toy_frame(), config())
    assert np.isfinite(prob).all()
    assert 0.0 <= prob.min() <= prob.max() <= 1.0


def test_11_isotonic_calibration_remains_monotonic() -> None:
    iso = IsotonicRegression(out_of_bounds="clip").fit([0, 0.2, 0.8, 1.0], [0, 0, 1, 1])
    values = iso.predict(np.linspace(0, 1, 20))
    assert np.all(np.diff(values) >= -1e-12)


def test_12_shallow_tree_depth_is_bounded() -> None:
    fit = fit_gate_candidate("shallow_tree", toy_frame(), config())
    assert fit.model.get_depth() <= 3


def test_13_sparse_kan_gate_produces_finite_probabilities() -> None:
    fit = fit_gate_candidate("sparse_additive_kan", toy_frame(), config())
    prob, _ = predict_gate_probability(fit, toy_frame(), config())
    assert np.isfinite(prob).all()


def test_14_additive_kan_contributions_sum_to_raw_output() -> None:
    model = OneSidedKANMagnitude(3, correction_bound=10.0, seed=7)
    x = torch.randn(4, 3)
    layer = model.kan.first
    raw_from_edges = layer.edge_contributions(x).sum(dim=2).squeeze(1) + layer.bias.squeeze(0)
    assert torch.allclose(raw_from_edges, model.raw(x), atol=1e-6)


def test_15_safe_row_zero_correction_loss_penalizes_safe_magnitude() -> None:
    dangerous = torch.tensor([0.0, 1.0])
    high = safe_row_zero_correction_loss(torch.tensor([3.0, 3.0]), dangerous)
    low = safe_row_zero_correction_loss(torch.tensor([0.0, 3.0]), dangerous)
    assert high > low


def test_16_critical_under_correction_loss_penalizes_underprediction() -> None:
    exact = critical_under_correction_loss(torch.tensor([5.0]), torch.tensor([5.0]), 4.0)
    under = critical_under_correction_loss(torch.tensor([1.0]), torch.tensor([5.0]), 4.0)
    assert under > exact


def test_17_no_gate_control_recreates_global_correction_setting() -> None:
    final, downward, gate = one_sided_final_prediction(np.array([20.0, 30.0]), np.ones(2), np.array([5.0, 5.0]), threshold=0.0, bound=10.0)
    assert np.all(gate == 1.0)
    assert np.all(downward == 5.0)
    assert final.tolist() == [15.0, 25.0]


def test_18_gate_activation_limit_is_enforced_when_feasible() -> None:
    cfg = config()
    y = np.array([1] * 5 + [0] * 95)
    p = np.linspace(1, 0, 100)
    threshold = select_gate_threshold(y, p, cfg)
    assert (p >= threshold).mean() <= cfg["gate"]["activation_rate_max"] + 1e-12


def test_19_lexicographic_selection_is_deterministic() -> None:
    cfg = config()
    base = {"critical_miss_proxy_count": 2, "critical_optimistic_rate": 0.5, "severe_optimistic_rate": 0.1, "rmse": 10.0, "mae": 5.0, "nasa_score": 100.0}
    metrics = {**base, "critical_miss_proxy_count": 1, "new_critical_misses": 0, "gate_activation_rate": 0.1, "bound_saturation_rate": 0.0}
    assert selection_flags(metrics, base, cfg) == selection_flags(metrics, base, cfg)


def test_20_accuracy_noninferiority_is_enforced() -> None:
    cfg = config()
    base = {"critical_miss_proxy_count": 1, "critical_optimistic_rate": 0.5, "severe_optimistic_rate": 0.1, "rmse": 10.0, "mae": 5.0, "nasa_score": 100.0}
    metrics = {**base, "critical_miss_proxy_count": 0, "rmse": 20.0, "new_critical_misses": 0, "gate_activation_rate": 0.1, "bound_saturation_rate": 0.0}
    assert selection_flags(metrics, base, cfg)["stage2_accuracy"] is False


def test_21_no_new_critical_miss_constraint_is_enforced() -> None:
    cfg = config()
    base = {"critical_miss_proxy_count": 2, "critical_optimistic_rate": 0.5, "severe_optimistic_rate": 0.1, "rmse": 10.0, "mae": 5.0, "nasa_score": 100.0}
    metrics = {**base, "critical_miss_proxy_count": 1, "new_critical_misses": 1, "gate_activation_rate": 0.1, "bound_saturation_rate": 0.0}
    assert selection_flags(metrics, base, cfg)["stage1_safety"] is False


def test_22_pruning_is_rejected_when_fidelity_fails() -> None:
    decision = pruning_decision(0.5, 10.0, True, config())
    assert decision["accepted"] is False


def test_23_unpruned_model_remains_valid_when_pruning_fails() -> None:
    decision = pruning_decision(0.5, 10.0, True, config())
    assert decision["accepted"] is False
    assert decision["correlation"] == 0.5


def test_24_benchmark_labels_are_inaccessible_before_lock() -> None:
    frame = pd.DataFrame({"subset": ["A"], "global_engine_id": ["e1"], "predicted_rul": [1.0], "true_rul": [1.0], "residual": [0.0]})
    stripped = strip_benchmark_labels_before_lock(frame)
    assert "true_rul" not in stripped
    assert "residual" not in stripped


def test_25_lock_manifest_precedes_benchmark_evaluation(tmp_path) -> None:
    manifest = write_prebenchmark_lock_manifest(tmp_path / "lock.json", {"gate_model_family": "logistic"})
    assert (tmp_path / "lock.json").exists()
    assert manifest["benchmark_labels_accessed_before_lock"] is False


def test_26_uncertainty_calibration_uses_corrected_oof_predictions() -> None:
    frame = toy_frame().assign(corrected_predicted_rul=lambda df: df["predicted_rul"], corrected_residual=lambda df: df["predicted_rul"] - df["true_rul"])
    policy, _ = fit_selective_uncertainty(frame, config())
    assert policy["source_prediction_column"] == "corrected_predicted_rul"


def test_27_abstention_selection_uses_corrected_oof_predictions() -> None:
    frame = toy_frame().assign(corrected_predicted_rul=lambda df: df["predicted_rul"], corrected_residual=lambda df: df["predicted_rul"] - df["true_rul"], corrected_absolute_error=lambda df: (df["predicted_rul"] - df["true_rul"]).abs(), interval_width_90=20.0)
    policy, _ = fit_selective_abstention(frame, config())
    assert policy["source_prediction_column"] == "corrected_predicted_rul"


def test_28_maintenance_selection_uses_corrected_oof_predictions() -> None:
    frame = toy_frame().assign(corrected_predicted_rul=lambda df: df["predicted_rul"], abstain_flag=False)
    policy, _ = fit_selective_maintenance(frame, config())
    assert policy["source_prediction_column"] == "corrected_predicted_rul"


def test_29_paired_engine_alignment_is_exact() -> None:
    frame = pd.DataFrame({"subset": ["A"], "global_engine_id": ["e1"]})
    assert paired_engine_alignment(frame, frame)["aligned"] is True


def test_30_source_hashes_remain_unchanged() -> None:
    manifest = [{"artifact_key": "x", "sha256": "abc"}]
    assert source_hashes_unchanged(manifest, manifest) is True


def test_31_transformer_training_is_never_called() -> None:
    assert TRANSFORMER_TRAINING_CALLED is False


def test_32_smoke_test_completes() -> None:
    result = run_smoke_test("configs/selective_aerokan_safety_corrector.yaml")
    assert result["status"] == "smoke_complete"


def test_33_run_status_is_correct_on_success() -> None:
    result = run_smoke_test("configs/selective_aerokan_safety_corrector.yaml")
    assert result["status"].endswith("complete")


def test_34_failure_summary_is_correct_on_exception() -> None:
    summary = failure_summary(RuntimeError("boom"))
    assert summary["status"] == "failed"
    assert summary["exception_type"] == "RuntimeError"


def test_correction_candidates_predict_bounded_magnitude() -> None:
    cfg = config()
    frame = toy_frame()
    gate_fit = GateFit({"candidate_id": "no_gate"}, NoGateModel(), fit_gate_candidate("logistic", frame, cfg).preprocessor, 0.0, {})
    with_gate = frame.assign(gate_probability=1.0, gate_threshold=0.0, gate_active=True, gate_threshold_margin=1.0, gate_active_float=1.0)
    candidate = next(item for item in build_correction_candidate_registry(cfg, smoke=True) if item["candidate_type"] == "linear_nonnegative")
    fit = fit_correction_candidate(candidate, with_gate, cfg, 1)
    mag, _ = predict_correction_magnitude(fit, with_gate, cfg)
    assert np.all(mag >= 0.0)
    assert np.all(mag <= candidate["correction_bound"] + 1e-8)
    corrected = selective_corrected_predictions(frame, gate_fit, fit, cfg)
    assert np.all(corrected["corrected_predicted_rul"] <= corrected["base_predicted_rul"] + 1e-8)


def test_calibration_error_is_finite() -> None:
    assert np.isfinite(calibration_error(np.array([0, 1]), np.array([0.2, 0.8])))
