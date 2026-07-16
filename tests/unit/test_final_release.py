from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aeroguard.inference.artifact_loader import load_manifest, validate_component_hashes
from aeroguard.inference.monitoring import INFERENCE_LOG_FIELDS, monitoring_spec, schema_hash
from aeroguard.inference.predictor import AeroGuardPredictor
from aeroguard.inference.safety_guard import apply_critical_boundary_guard
from aeroguard.inference.schemas import OUTPUT_FIELDS, REQUIRED_INPUT_COLUMNS


MANIFEST = ROOT / "artifacts" / "final_release" / "frozen_system_manifest.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_history() -> pd.DataFrame:
    return pd.read_csv(ROOT / "examples" / "sample_engine_history.csv")


def test_model_registry_discovers_expected_candidates() -> None:
    registry = pd.read_csv(ROOT / "reports" / "final_release" / "model_registry.csv")
    assert len(registry) >= 64
    expected = {
        "classical_random_forest",
        "patch_transformer_10x5_mean_b",
        "physics_regime",
        "two_layer_compact_kan_h8",
        "selective_one_sided_aerokan_safety_corrector",
        "critical_boundary_safety_guarded_transformer",
    }
    assert expected.issubset(set(registry["model_id"]))


def test_exact_model_ids_map_to_correct_display_names() -> None:
    mapping = pd.read_csv(ROOT / "reports" / "final_release" / "model_name_mapping.csv").set_index("model_id")
    assert mapping.loc["patch_transformer_10x5_mean_b", "display_name"] == "Patch Transformer — 10×5 Patches with Mean Pooling"
    assert mapping.loc["physics_regime", "display_name"] == "Regime-Consistent Physics-Guided Patch Transformer"
    assert mapping.loc["two_layer_compact_kan_h8", "display_name"] == "AeroKAN-PHM Compact Residual Corrector"
    assert mapping.loc["critical_boundary_safety_guarded_transformer", "display_name"] == "Critical-Boundary Safety-Guarded Physics-Guided Transformer"


def test_no_public_comparison_uses_forbidden_baseline_phrase() -> None:
    forbidden = "repository-derived " + "baseline"
    for rel in [
        "reports/final_release/headline_model_comparison.csv",
        "reports/final_release/point_prediction_comparison.csv",
        "reports/final_release/fixed_policy_safety_comparison.csv",
        "README.md",
        "MODEL_CARD.md",
    ]:
        assert forbidden not in (ROOT / rel).read_text(encoding="utf-8").lower()


def test_phase_labels_appear_only_as_registry_provenance() -> None:
    registry = pd.read_csv(ROOT / "reports" / "final_release" / "model_registry.csv")
    public_tables = ["headline_model_comparison.csv", "point_prediction_comparison.csv", "fixed_policy_safety_comparison.csv"]
    for filename in public_tables:
        text = (ROOT / "reports" / "final_release" / filename).read_text(encoding="utf-8")
        assert ("Phase 5C " + "model") not in text
        assert ("Phase 5D " + "model") not in text
    assert registry["development_stage"].str.contains("Phase", na=False).any()


def test_point_comparison_uses_aligned_engine_keys() -> None:
    aligned = pd.read_csv(ROOT / "reports" / "final_release" / "aligned_benchmark_predictions.csv")
    assert len(aligned) == 707
    assert aligned[["subset", "global_engine_id", "final_observed_cycle"]].duplicated().sum() == 0
    point = pd.read_csv(ROOT / "reports" / "final_release" / "point_prediction_comparison.csv")
    assert set(point["engine_count"]) == {707}
    assert set(point["true_rul_definition"]) == {"uncapped"}


def test_fixed_policy_comparison_uses_identical_policy() -> None:
    fixed = pd.read_csv(ROOT / "reports" / "final_release" / "fixed_policy_safety_comparison.csv")
    assert set(fixed["policy_id"]) == {"point_u15_s30_i60"}
    assert fixed["fixed_policy_label"].nunique() == 1
    assert set(fixed["abstain_review_count"]) == {0}


def test_native_policy_comparison_is_labelled() -> None:
    native = pd.read_csv(ROOT / "reports" / "final_release" / "native_system_comparison.csv")
    assert native["native_policy_label"].str.contains("native locked uncertainty").all()


def test_validation_only_metrics_are_not_benchmark_metrics() -> None:
    validation = pd.read_csv(ROOT / "reports" / "final_release" / "validation_candidate_comparison.csv")
    assert not validation.empty
    assert validation["benchmark_evaluated"].eq(False).all()


def test_missing_metrics_remain_missing() -> None:
    registry = pd.read_csv(ROOT / "reports" / "final_release" / "model_registry.csv")
    classical = registry[registry["model_id"] == "classical_random_forest"].iloc[0]
    assert pd.isna(classical["parameter_count"])


def test_final_manifest_contains_required_components() -> None:
    manifest = load_manifest(MANIFEST)
    for component in [
        "physics_regime_checkpoint",
        "final_preprocessor",
        "critical_boundary_guard",
        "uncertainty_model",
        "maintenance_policy",
        "metric_definition_registry",
    ]:
        assert component in manifest["components"]


def test_manifest_hashes_validate() -> None:
    manifest = load_manifest(MANIFEST)
    assert validate_component_hashes(manifest) == []


def test_feature_schema_hash_validates() -> None:
    manifest = load_manifest(MANIFEST)
    digest = hashlib.sha256(json.dumps(manifest["feature_schema"], sort_keys=True).encode("utf-8")).hexdigest()
    assert digest == manifest["feature_schema_hash"]


def test_predictor_loads_without_training() -> None:
    predictor = AeroGuardPredictor.from_manifest(MANIFEST)
    assert predictor.model_version == "aeroguard-phm-safety-v1"


def test_predictor_rejects_missing_columns() -> None:
    predictor = AeroGuardPredictor.from_manifest(MANIFEST)
    result = predictor.predict_engine(sample_history().drop(columns=["sensor_1"]))
    assert result["valid"] is False
    assert any(error["code"] == "missing_column" for error in result["errors"])


def test_predictor_rejects_duplicate_cycles() -> None:
    predictor = AeroGuardPredictor.from_manifest(MANIFEST)
    frame = sample_history()
    frame.loc[1, "cycle"] = frame.loc[0, "cycle"]
    result = predictor.predict_engine(frame)
    assert result["valid"] is False
    assert any(error["code"] == "duplicate_cycles" for error in result["errors"])


def test_predictor_handles_short_sequences() -> None:
    predictor = AeroGuardPredictor.from_manifest(MANIFEST)
    result = predictor.predict_engine(sample_history().head(5))
    assert result["valid"] is True
    assert any(warning["code"] == "short_history" for warning in result["warnings"])


def test_predictor_outputs_non_negative_rul() -> None:
    predictor = AeroGuardPredictor.from_manifest(MANIFEST)
    result = predictor.predict_engine(sample_history())
    assert result["base_rul"] >= 0
    assert result["safety_adjusted_rul"] >= 0


def test_safety_adjusted_rul_follows_frozen_guard() -> None:
    guarded = apply_critical_boundary_guard(20.0, boundary_low=15.0, boundary_high=25.0, margin=0.5, bound=10.0)
    assert guarded["safety_guard_activated"] is True
    assert guarded["correction_cycles"] == 5.5
    assert guarded["safety_adjusted_rul"] == 14.5


def test_output_schema_is_complete() -> None:
    prediction = read_json(ROOT / "examples" / "sample_prediction.json")
    for field in OUTPUT_FIELDS:
        assert field in prediction


def test_cli_inference_works_on_synthetic_data(tmp_path: Path) -> None:
    from aeroguard.inference.cli import run_cli

    output = tmp_path / "prediction.json"
    code = run_cli([
        "--manifest",
        str(MANIFEST),
        "--input",
        str(ROOT / "examples" / "sample_engine_history.csv"),
        "--output",
        str(output),
    ])
    assert code == 0
    assert read_json(output)["model_version"] == "aeroguard-phm-safety-v1"


def test_batch_inference_preserves_engine_identity() -> None:
    predictor = AeroGuardPredictor.from_manifest(MANIFEST)
    first = sample_history()
    second = sample_history()
    first["engine_id"] = "engine-a"
    second["engine_id"] = "engine-b"
    results = predictor.predict_batch([first, second])
    assert [item["engine_id"] for item in results] == ["engine-a", "engine-b"]


def test_api_module_is_import_safe() -> None:
    import aeroguard.api.app as api_app

    assert hasattr(api_app, "app")


def test_dashboard_module_is_import_safe() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("dashboard_app", ROOT / "dashboard" / "app.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "load_predictor")


def test_monitoring_log_schema_is_valid() -> None:
    spec = monitoring_spec()
    assert "safety_guard_activation_rate" in spec["components"]
    assert set(INFERENCE_LOG_FIELDS).issubset(set(spec["inference_log_fields"]))
    assert len(schema_hash(REQUIRED_INPUT_COLUMNS)) == 64


def test_source_artifact_hashes_remain_unchanged() -> None:
    manifest = load_manifest(MANIFEST)
    for rel_path, record in manifest["source_artifact_manifest"].items():
        path = ROOT / rel_path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["sha256"]


def test_previous_outputs_are_not_marked_modified() -> None:
    summary = read_json(ROOT / "reports" / "final_release" / "release_summary.json")
    assert summary["previous_outputs_modified"] is False


def test_readme_table_matches_generated_csv() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    headline = pd.read_csv(ROOT / "reports" / "final_release" / "headline_model_comparison.csv")
    for model in headline["Model"]:
        assert model in readme


def test_model_card_metrics_match_authoritative_reports() -> None:
    model_card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    final = pd.read_csv(ROOT / "reports" / "final_release" / "point_prediction_comparison.csv")
    rmse = final.loc[final["model_id"] == "critical_boundary_safety_guarded_transformer", "rmse"].iloc[0]
    assert f"{rmse:.4f}" in model_card


def test_reproducibility_commands_are_valid() -> None:
    text = (ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    assert "aeroguard.pipelines.build_final_release" in text
    assert "pytest" in text


def test_no_training_function_called_or_imported_by_release_builder() -> None:
    source = (ROOT / "src" / "aeroguard" / "pipelines" / "build_final_release.py").read_text(encoding="utf-8")
    assert "train_model" not in source
    assert "train_fixed_epochs" not in source


def test_no_package_installation_recorded() -> None:
    summary = read_json(ROOT / "reports" / "final_release" / "release_summary.json")
    assert summary["packages_installed"] is False


def test_optional_dependency_status_recorded() -> None:
    summary = read_json(ROOT / "reports" / "final_release" / "release_summary.json")
    assert summary["optional_dependencies"]["fastapi"] is True
    assert summary["optional_dependencies"]["streamlit"] is True


def test_final_smoke_outputs_exist() -> None:
    required = [
        ROOT / "reports" / "final_release" / "model_registry.csv",
        ROOT / "reports" / "final_release" / "point_prediction_comparison.csv",
        ROOT / "artifacts" / "final_release" / "frozen_system_manifest.json",
        ROOT / "examples" / "sample_prediction.json",
    ]
    assert all(path.exists() for path in required)
