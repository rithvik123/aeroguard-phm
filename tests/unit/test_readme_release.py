from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
ARCHIVE = ROOT / "docs" / "archive" / "README_before_detailed_release.md"
IMAGE_PLAN = ROOT / "docs" / "assets" / "readme" / "README_IMAGE_PLAN.md"

SPEC = importlib.util.spec_from_file_location("validate_readme", ROOT / "scripts" / "validate_readme.py")
assert SPEC is not None and SPEC.loader is not None
validate_readme = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validate_readme)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_by(rows: list[dict[str, str]], column: str, value: str) -> dict[str, str]:
    for row in rows:
        if row[column] == value:
            return row
    raise AssertionError(f"Missing row where {column}={value}")


def visible_readme() -> str:
    return validate_readme.strip_html_comments(read_text(README))


def test_01_readme_and_archive_exist() -> None:
    assert README.exists()
    assert ARCHIVE.exists()
    assert "Phase 6" in read_text(ARCHIVE)


def test_02_readme_is_detailed_release_document() -> None:
    words = re.findall(r"\b\w+\b", read_text(README))
    assert len(words) >= 4000
    assert "Frozen v1.0.0 release" in read_text(README)


def test_03_required_top_level_sections_are_present() -> None:
    headings = validate_readme.visible_headings(read_text(README))
    for section in validate_readme.REQUIRED_SECTIONS:
        assert section in headings


def test_04_quick_navigation_anchors_resolve() -> None:
    errors = [error for error in validate_readme.validate(ROOT) if "missing anchor" in error]
    assert errors == []


def test_05_public_filesystem_links_resolve() -> None:
    errors = [error for error in validate_readme.validate(ROOT) if "missing file" in error]
    assert errors == []


def test_06_release_snapshot_uses_final_system_names() -> None:
    text = read_text(README)
    assert validate_readme.FINAL_SYSTEM in text
    assert validate_readme.FINAL_MODEL in text
    assert validate_readme.BACKBONE in text
    assert "aeroguard-phm-safety-v1" in text


def test_07_headline_metrics_match_frozen_csv() -> None:
    text = read_text(README)
    rows = read_rows(ROOT / "reports" / "final_release" / "headline_model_comparison.csv")
    final = row_by(rows, "Status", "Final selected system")
    assert f"{float(final['Overall MAE']):.4f}" in text
    assert f"{float(final['Overall RMSE']):.4f}" in text
    assert f"{float(final['Severe optimism']):.4f}" in text
    assert f"{float(final['Operational recall']):.4f}" in text
    assert f"{float(final['Review workload']):.4f}" in text
    assert f"| Critical misses | {int(float(final['Critical misses']))} |" in text


def test_08_model_registry_candidate_count_is_reported() -> None:
    rows = read_rows(ROOT / "reports" / "final_release" / "model_registry.csv")
    assert len(rows) == 64
    assert f"| Model/system candidates evaluated | {len(rows)} |" in read_text(README)


def test_09_improvement_over_patch_transformer_matches_csvs() -> None:
    text = read_text(README)
    headline = read_rows(ROOT / "reports" / "final_release" / "headline_model_comparison.csv")
    fixed = read_rows(ROOT / "reports" / "final_release" / "fixed_policy_safety_comparison.csv")
    final_headline = row_by(headline, "Status", "Final selected system")
    patch_headline = row_by(headline, "Family", "Patch Transformer")
    final_policy = row_by(fixed, "model_id", "critical_boundary_safety_guarded_transformer")
    patch_policy = row_by(fixed, "model_id", "patch_transformer_10x5_mean_b")
    assert f"{float(final_headline['Overall MAE']) - float(patch_headline['Overall MAE']):+.4f} cycles" in text
    assert f"{float(final_headline['Overall RMSE']) - float(patch_headline['Overall RMSE']):+.4f} cycles" in text
    assert f"{int(float(final_policy['critical_miss_count'])) - int(float(patch_policy['critical_miss_count'])):+d}" in text
    assert f"{float(final_policy['operational_recall']) - float(patch_policy['operational_recall']):+.4f}" in text
    assert f"{float(final_policy['review_workload']) - float(patch_policy['review_workload']):+.4f}" in text


def test_10_final_guard_rule_is_auditable_and_downward_only() -> None:
    text = read_text(README)
    assert "15 < base_rul <= 25" in text
    assert "base_rul - min(correction_bound, base_rul - (urgent_threshold - margin))" in text
    assert "correction_bound: 10.0 cycles" in text
    assert "0.5 cycles" in text
    assert "makes no positive RUL corrections" in text


def test_11_kan_is_explained_but_not_claimed_as_deployed() -> None:
    text = read_text(README)
    assert "AeroKAN-PHM Compact Residual Corrector" in text
    assert "Selective One-Sided AeroKAN Safety Corrector" in text
    assert "not a learned KAN component" in text
    assert "KAN experiments were not selected for deployment" in text


def test_12_dataset_subsets_and_pipeline_are_documented() -> None:
    text = read_text(README)
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        assert subset in text
    for token in ["operating settings", "sensor channels", "50-cycle", "past-only"]:
        assert token in text


def test_13_uncertainty_radii_are_reported() -> None:
    text = read_text(README)
    for token in ["32.1584", "53.5571", "86.5394", "global split conformal"]:
        assert token in text


def test_14_maintenance_policy_thresholds_are_reported() -> None:
    text = read_text(README)
    for token in ["RUL <= 15", "15 < RUL <= 30", "30 < RUL <= 60", "RUL > 60"]:
        assert token in text
    for action in ["URGENT_ENGINEERING_REVIEW", "SCHEDULE_MAINTENANCE", "PLAN_INSPECTION", "CONTINUE_MONITORING"]:
        assert action in text


def test_15_api_endpoints_are_documented() -> None:
    text = read_text(README)
    for endpoint in ["GET /health", "GET /model", "POST /validate-input", "POST /predict", "POST /predict-batch"]:
        assert endpoint in text


def test_16_streamlit_dashboard_and_commented_gallery_are_documented() -> None:
    text = read_text(README)
    assert "dashboard/app.py" in text
    assert "SCREENSHOT GALLERY TO ENABLE AFTER REAL SCREENSHOTS ARE ADDED" in text
    for image_path in validate_readme.EXPECTED_IMAGE_PATHS:
        if "streamlit_" in image_path:
            assert image_path in text


def test_17_production_inference_example_has_no_true_rul_label() -> None:
    section = validate_readme.section_text(read_text(README), "Production Inference Output")
    assert "true_rul" not in section
    assert "true RUL" not in section
    assert "ground-truth" not in section.lower()
    assert '"safety_adjusted_rul"' in section
    assert '"maintenance_action"' in section


def test_18_no_private_paths_or_unrelated_baseline_terms_are_visible() -> None:
    text = visible_readme()
    assert not validate_readme.PRIVATE_PATH_PATTERN.search(text)
    for phrase in validate_readme.FORBIDDEN_PUBLIC_PHRASES:
        assert phrase not in text.lower()


def test_19_image_plan_lists_all_expected_assets_without_rendering_missing_images() -> None:
    assert IMAGE_PLAN.exists()
    readme_text = read_text(README)
    plan_text = read_text(IMAGE_PLAN)
    visible_images = validate_readme.visible_local_images(readme_text)
    assert set(visible_images) == {
        "docs/assets/readme/architecture/model_development_journey.png",
        "docs/assets/readme/architecture/critical_boundary_safety_guard.png",
    }
    for image_path in visible_images:
        assert (ROOT / image_path).exists()
    for image_path in validate_readme.EXPECTED_IMAGE_PATHS:
        assert image_path in readme_text
        assert image_path in plan_text


def test_20_readme_validator_and_frozen_source_hashes_pass() -> None:
    assert validate_readme.validate(ROOT) == []
    manifest = json.loads((ROOT / "artifacts" / "final_release" / "frozen_system_manifest.json").read_text(encoding="utf-8"))
    for rel_path, metadata in manifest["source_artifact_manifest"].items():
        path = ROOT / rel_path
        assert path.exists()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == metadata["sha256"]
