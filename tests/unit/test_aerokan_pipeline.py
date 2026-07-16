from __future__ import annotations

import numpy as np
import pandas as pd

from aeroguard.pipelines.train_aerokan_rul_corrector import (
    apply_maintenance,
    build_candidate_registry,
    build_named_features,
    candidate_feature_names,
    engine_balanced_weights,
    fit_feature_preprocessor,
    load_config,
    point_metrics,
    residual_target,
    run_dry_run,
    run_smoke_test,
    safety_state,
    split_by_engine,
    synthetic_frame,
    transform_feature_frame,
)


def config() -> dict:
    return load_config("configs/aerokan_rul_corrector.yaml")


def test_named_feature_order_and_no_forbidden_tokens() -> None:
    names = candidate_feature_names(config())
    assert names == candidate_feature_names(config())
    assert len(names) > 20
    assert not any("true_rul" in name or "target" in name for name in names)


def test_feature_extraction_and_preprocessor_use_training_frame_only() -> None:
    cfg = config()
    predictions, sensors = synthetic_frame(n_engines=6, windows=3)
    features = build_named_features(predictions, sensors, cfg)
    dev, val, split = split_by_engine(features, 0.5, 99)
    prep = fit_feature_preprocessor(dev, cfg)
    x_dev, _ = transform_feature_frame(dev, prep, cfg)
    x_val, _ = transform_feature_frame(val, prep, cfg)
    assert split["engine_overlap_count"] == 0
    assert x_dev.shape[1] == len(prep["feature_names"])
    assert x_val.shape[1] == len(prep["feature_names"])
    assert prep["healthy_row_definition"].startswith("true_rul")


def test_residual_orientation_and_engine_balanced_weights() -> None:
    frame = pd.DataFrame(
        {
            "subset": ["A", "A", "A"],
            "global_engine_id": ["e1", "e1", "e2"],
            "true_rul": [10.0, 20.0, 30.0],
            "predicted_rul": [15.0, 10.0, 35.0],
        }
    )
    assert residual_target(frame).tolist() == [-5.0, 10.0, -5.0]
    weights = engine_balanced_weights(frame)
    assert np.isclose(weights[:2].sum(), weights[2], atol=1e-6)


def test_point_metrics_and_maintenance_distinguish_direct_and_operational() -> None:
    frame = pd.DataFrame(
        {
            "subset": ["A", "A", "A"],
            "global_engine_id": ["e1", "e2", "e3"],
            "true_rul": [10.0, 12.0, 80.0],
            "predicted_rul": [12.0, 50.0, 70.0],
            "corrected_predicted_rul": [12.0, 50.0, 70.0],
            "abstain_flag": [False, True, False],
        }
    )
    scored = apply_maintenance(frame, {"urgent_threshold": 15.0, "schedule_threshold": 30.0, "inspection_threshold": 60.0})
    assert scored.loc[1, "maintenance_action"] == "ABSTAIN_AND_REVIEW"
    metrics = point_metrics(frame, frame["corrected_predicted_rul"].to_numpy())
    assert metrics["critical_miss_proxy_count"] == 1
    assert safety_state(pd.Series([15, 16, 30, 31, 60, 61, 90, 91])).tolist() == [
        "CRITICAL",
        "NEAR_TERM",
        "NEAR_TERM",
        "INSPECTION_WINDOW",
        "INSPECTION_WINDOW",
        "MONITORING",
        "MONITORING",
        "HEALTHY",
    ]


def test_candidate_registry_and_dry_run() -> None:
    candidates = build_candidate_registry(config())
    assert {"phase5c_frozen_baseline", "linear_ridge_residual", "small_mlp_residual", "safety_weighted_sparse_kan"}.issubset({c["candidate_id"] for c in candidates})
    dry = run_dry_run("configs/aerokan_rul_corrector.yaml")
    assert dry["status"] == "dry_run_complete"
    assert dry["benchmark_labels_excluded_from_selection"] is True


def test_smoke_pipeline_no_benchmark_leakage() -> None:
    result = run_smoke_test("configs/aerokan_rul_corrector.yaml")
    assert result["status"] == "smoke_complete"
    assert result["benchmark_leakage"] is False
    assert result["backbone_training_called"] is False
