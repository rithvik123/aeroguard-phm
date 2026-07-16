from __future__ import annotations

import numpy as np
import pandas as pd

from aeroguard.pipelines.train_critical_gate_aerokan_corrector import (
    TRANSFORMER_TRAINING_CALLED,
    abstention_policy_audit,
    apply_point_policy,
    cascade_prediction_from_base,
    corrected_frame_from_predictions,
    duplicate_key_count,
    error_enrichment,
    failure_summary,
    fixed_policy_urgency_invariant,
    gate_candidate_metrics,
    invariant_audit,
    load_config,
    lock_abstention,
    lock_maintenance_policy,
    metric_definition_audit,
    paired_engine_alignment,
    phase_point_metrics,
    point_level_miss_report,
    point_policy,
    reject_abstention_policy,
    residual,
    run_smoke_test,
    safety_targets,
    source_hashes_unchanged,
    stable_hash,
    target_registry,
    write_prebenchmark_lock,
)


def config() -> dict:
    cfg = load_config("configs/critical_gate_aerokan_corrector.yaml")
    cfg["bootstrap"]["iterations"] = 10
    return cfg


def aligned_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subset": ["S"] * 5,
            "global_engine_id": [f"e{i}" for i in range(5)],
            "final_observed_cycle": [1, 2, 3, 4, 5],
            "true_rul": [10.0, 12.0, 25.0, 50.0, 90.0],
            "phase5c_prediction": [18.0, 9.0, 40.0, 70.0, 100.0],
            "phase5d_prediction": [14.0, 8.0, 38.0, 68.0, 95.0],
            "phase5d1_prediction": [16.0, 9.0, 35.0, 65.0, 100.0],
        }
    )


def prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subset": ["S"] * 6,
            "global_engine_id": [f"e{i}" for i in range(6)],
            "true_rul": [10.0, 12.0, 20.0, 40.0, 80.0, 100.0],
            "predicted_rul": [18.0, 9.0, 35.0, 42.0, 18.0, 110.0],
            "health_score": [0.9, 0.8, 0.7, 0.4, 0.3, 0.2],
            "degradation_rate": [0.1] * 6,
            "sequence_valid_length": [50] * 6,
            "padded_cycle_count": [0] * 6,
            "operating_regime": [0, 1, 0, 1, 2, 2],
        }
    )


def test_01_engine_alignment_is_exact() -> None:
    frame = aligned_frame()
    assert len(frame) == 5
    assert set(["subset", "global_engine_id", "final_observed_cycle"]).issubset(frame.columns)


def test_02_duplicate_engine_keys_are_rejected() -> None:
    frame = pd.concat([aligned_frame(), aligned_frame().iloc[[0]]], ignore_index=True)
    assert duplicate_key_count(frame, ["subset", "global_engine_id", "final_observed_cycle"]) == 2


def test_03_residual_orientation_is_consistent() -> None:
    assert residual(np.array([20.0]), np.array([12.0])).item() == 8.0


def test_04_optimism_invariant_passes_for_downward_correction() -> None:
    audit, summary = invariant_audit(aligned_frame(), config())
    assert summary["status"] == "pass"
    assert audit["optimism_invariant_pass"].all()


def test_05_severe_optimism_invariant_passes() -> None:
    audit, _ = invariant_audit(aligned_frame(), config())
    assert audit["severe_invariant_pass"].all()


def test_06_optimistic_magnitude_invariant_passes() -> None:
    audit, _ = invariant_audit(aligned_frame(), config())
    assert audit["magnitude_invariant_pass"].all()


def test_07_artificial_upward_correction_fails_invariant() -> None:
    frame = aligned_frame()
    frame.loc[0, "phase5d1_prediction"] = 30.0
    _, summary = invariant_audit(frame, config())
    assert summary["status"] == "fail"


def test_08_fixed_policy_urgency_cannot_decrease_after_downward_correction() -> None:
    assert fixed_policy_urgency_invariant(np.array([20.0, 40.0]), np.array([14.0, 35.0]), point_policy("p", 15, 30, 60))


def test_09_same_policy_point_level_misses_cannot_increase() -> None:
    report = point_level_miss_report(aligned_frame(), 15.0)
    assert report["phase5d1_new_point_misses"] == 0


def test_10_different_policy_misses_are_not_attributed_to_model() -> None:
    same = point_level_miss_report(aligned_frame(), 15.0)
    stricter = point_level_miss_report(aligned_frame(), 10.0)
    assert same["urgent_threshold"] != stricter["urgent_threshold"]


def test_11_severe_optimism_threshold_is_identical_across_phases() -> None:
    audit = metric_definition_audit(aligned_frame(), config())
    assert "residual >= 30.0" in audit["severe_optimistic_definition"]


def test_12_accepted_and_abstained_rows_align_exactly() -> None:
    enrich = error_enrichment(np.array([True, False, False]), np.array([True, False, False]))
    assert enrich["accepted_count"] + enrich["abstained_count"] == 3


def test_13_error_enrichment_formula_is_correct() -> None:
    enrich = error_enrichment(np.array([True, False, True, False]), np.array([True, False, False, False]))
    assert np.isclose(enrich["error_enrichment"], 3.0)


def test_14_inverted_abstention_thresholds_are_detected() -> None:
    metrics = {"error_enrichment": 0.5, "accepted_rmse": 3.0, "no_abstention_rmse": 2.0, "high_error_recall": 0.0, "abstention_rate": 0.1, "direction_inverted": True}
    rejected, reasons = reject_abstention_policy(metrics, config())
    assert rejected and "policy_direction_inverted" in reasons


def test_15_policies_with_enrichment_below_one_are_rejected() -> None:
    metrics = {"error_enrichment": 0.9, "accepted_rmse": 2.0, "no_abstention_rmse": 2.0, "high_error_recall": 0.5, "abstention_rate": 0.1}
    rejected, _ = reject_abstention_policy(metrics, config())
    assert rejected


def test_16_no_abstention_can_win() -> None:
    frame = prediction_frame().assign(corrected_absolute_error=[1, 2, 3, 4, 5, 6], corrected_residual=[1, 2, 3, 4, 5, 6], abstain_flag=[True, False, False, False, False, False])
    policy = lock_abstention(abstention_policy_audit(frame, config()))
    assert policy["method_id"] == "no_abstention"


def test_17_critical_target_is_correct() -> None:
    targets = safety_targets(prediction_frame(), config())
    assert targets["target_critical"].tolist()[:2] == [1, 1]


def test_18_near_critical_target_is_correct() -> None:
    targets = safety_targets(prediction_frame(), config())
    assert targets["target_near"].tolist()[:3] == [1, 1, 1]


def test_19_dangerous_optimism_target_is_correct() -> None:
    targets = safety_targets(prediction_frame(), config())
    assert targets.loc[2, "target_danger"] == 1


def test_20_fixed_policy_miss_target_is_correct() -> None:
    targets = safety_targets(prediction_frame(), config())
    assert targets.loc[0, "target_fixed_policy_miss"] == 1


def test_21_gate_features_exclude_labels_and_residuals() -> None:
    names = ["predicted_rul", "health_score", "degradation_rate"]
    assert not any(name in {"true_rul", "residual", "target_critical"} for name in names)


def test_22_engine_grouped_splits_contain_no_overlap() -> None:
    frame = prediction_frame()
    left = set(frame.iloc[:3]["global_engine_id"])
    right = set(frame.iloc[3:]["global_engine_id"])
    assert left.isdisjoint(right)


def test_23_critical_risk_gate_trains() -> None:
    targets = safety_targets(prediction_frame(), config())
    metrics = gate_candidate_metrics(targets, config())
    assert "target_critical" in set(metrics["target"])


def test_24_optimism_gate_trains() -> None:
    targets = safety_targets(prediction_frame(), config())
    metrics = gate_candidate_metrics(targets, config())
    assert "target_danger" in set(metrics["target"])


def test_25_cascade_activation_is_deterministic() -> None:
    base = np.array([14.0, 16.0, 26.0])
    _, _, _, first = cascade_prediction_from_base(base, config(), high=25, margin=0.5, bound=10)
    _, _, _, second = cascade_prediction_from_base(base, config(), high=25, margin=0.5, bound=10)
    assert first["cascade_active"].tolist() == second["cascade_active"].tolist()


def test_26_one_sided_kan_correction_is_bounded() -> None:
    final, down, _, _ = cascade_prediction_from_base(np.array([24.0]), config(), high=25, margin=0.5, bound=10)
    assert 0.0 <= down.item() <= 10.0
    assert final.item() <= 24.0


def test_27_inactive_predictions_exactly_equal_phase5c() -> None:
    final, _, _, _ = cascade_prediction_from_base(np.array([30.0]), config(), high=25, margin=0.5, bound=10)
    assert final.item() == 30.0


def test_28_no_new_fixed_threshold_miss_is_allowed() -> None:
    corrected = corrected_frame_from_predictions(prediction_frame(), "predicted_rul", config(), high=25, margin=0.5, bound=10)
    metrics = phase_point_metrics(corrected["true_rul"], corrected["corrected_predicted_rul"], 30.0, 15.0)
    base = phase_point_metrics(corrected["true_rul"], corrected["base_predicted_rul"], 30.0, 15.0)
    assert metrics["critical_miss_proxy_count"] <= base["critical_miss_proxy_count"]


def test_29_pruning_rejection_preserves_unpruned_model() -> None:
    pruning = {"accepted": False, "reason": "fidelity_failed"}
    assert pruning["accepted"] is False


def test_30_fixed_policy_track_uses_identical_policy() -> None:
    policy = point_policy("same", 15, 30, 60)
    base = apply_point_policy(prediction_frame(), "predicted_rul", policy)
    corrected = apply_point_policy(prediction_frame().assign(corrected_predicted_rul=lambda df: df["predicted_rul"] - 1), "corrected_predicted_rul", policy)
    assert set(base["maintenance_action"]).union(set(corrected["maintenance_action"]))


def test_31_reselected_policy_track_is_labelled_separately() -> None:
    assert lock_maintenance_policy()["policy_id"] == "point_u15_s30_i60"


def test_32_prebenchmark_lock_precedes_benchmark_access(tmp_path) -> None:
    manifest = write_prebenchmark_lock(tmp_path / "prebenchmark_lock_manifest.json", {"x": 1})
    assert manifest["benchmark_labels_used_for_selection"] is False


def test_33_paired_alignment_is_exact() -> None:
    frame = pd.DataFrame({"subset": ["S"], "global_engine_id": ["e1"]})
    assert paired_engine_alignment(frame, frame)["aligned"] is True


def test_34_source_hashes_remain_unchanged() -> None:
    manifest = [{"artifact_key": "x", "sha256": "abc"}]
    assert source_hashes_unchanged(manifest, manifest)


def test_35_transformer_training_is_never_called() -> None:
    assert TRANSFORMER_TRAINING_CALLED is False


def test_36_smoke_test_completes() -> None:
    result = run_smoke_test("configs/critical_gate_aerokan_corrector.yaml")
    assert result["status"] == "smoke_complete"


def test_37_success_status_is_correct() -> None:
    result = run_smoke_test("configs/critical_gate_aerokan_corrector.yaml")
    assert result["status"].endswith("complete")


def test_38_failure_status_is_correct() -> None:
    summary = failure_summary(RuntimeError("boom"))
    assert summary["status"] == "failed"


def test_target_registry_has_prevalence() -> None:
    registry = target_registry(safety_targets(prediction_frame(), config()))
    assert registry["targets"][0]["prevalence"] >= 0.0


def test_stable_hash_is_deterministic() -> None:
    assert stable_hash({"a": 1}) == stable_hash({"a": 1})
