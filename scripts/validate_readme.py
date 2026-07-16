from __future__ import annotations

import csv
import re
import sys
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
IMAGE_PLAN = ROOT / "docs" / "assets" / "readme" / "README_IMAGE_PLAN.md"
HEADLINE_CSV = ROOT / "reports" / "final_release" / "headline_model_comparison.csv"
FIXED_POLICY_CSV = ROOT / "reports" / "final_release" / "fixed_policy_safety_comparison.csv"
MODEL_REGISTRY_CSV = ROOT / "reports" / "final_release" / "model_registry.csv"

FINAL_MODEL = "Critical-Boundary Safety-Guarded Physics-Guided Transformer"
FINAL_SYSTEM = "AeroGuard-PHM Safety-Guarded RUL System"
BACKBONE = "Regime-Consistent Physics-Guided Patch Transformer"

REQUIRED_SECTIONS = [
    "Release Snapshot",
    "Quick Navigation",
    "What AeroGuard-PHM Solves",
    "Why RMSE Alone Is Not Enough",
    "Dataset: NASA C-MAPSS",
    "End-to-End Data Pipeline",
    "Models Evaluated",
    "Development Journey",
    "Final Architecture",
    "Physics-Guided Patch Transformer",
    "Final Safety Guard",
    "Final Results",
    "Improvement Over the Patch Transformer",
    "Subset Results",
    "Uncertainty Quantification",
    "Review and Maintenance Policy",
    "Interactive Streamlit Dashboard",
    "FastAPI Inference Service",
    "Quick Start",
    "Production Inference Output",
    "Monitoring",
    "Repository Structure",
    "Testing",
    "Reproducibility and Frozen Artifacts",
    "Limitations",
    "Future Work",
    "Responsible Use",
    "Citation and License",
]

REQUIRED_ASSET_DIRS = [
    "docs/assets/readme/hero",
    "docs/assets/readme/architecture",
    "docs/assets/readme/screenshots",
    "docs/assets/readme/charts",
]

EXPECTED_IMAGE_PATHS = [
    "docs/assets/readme/hero/aeroguard_phm_hero.png",
    "docs/assets/readme/hero/rul_problem_statement.png",
    "docs/assets/readme/architecture/model_development_journey.png",
    "docs/assets/readme/architecture/final_system_design.png",
    "docs/assets/readme/architecture/physics_guided_patch_transformer.png",
    "docs/assets/readme/architecture/critical_boundary_safety_guard.png",
    "docs/assets/readme/architecture/uncertainty_maintenance_flow.png",
    "docs/assets/readme/architecture/deployment_monitoring.png",
    "docs/assets/readme/screenshots/streamlit_01_home.png",
    "docs/assets/readme/screenshots/streamlit_02_input_validation.png",
    "docs/assets/readme/screenshots/streamlit_03_sensor_history.png",
    "docs/assets/readme/screenshots/streamlit_04_prediction_result.png",
    "docs/assets/readme/screenshots/streamlit_05_maintenance_explanation.png",
    "docs/assets/readme/screenshots/fastapi_swagger.png",
]

FORBIDDEN_PUBLIC_PHRASES = [
    "repository-derived " + "baseline",
    "polynomial color correction",
    "pcc baseline",
    "downloaded " + "repo",
    "original " + "repo model",
    "kan-gated",
    "kan-enhanced",
    "ground-truth change mask",
    "final deployed kan",
    "deployed kan model",
    "selected kan model",
]

PRIVATE_PATH_PATTERN = re.compile("(?i)(" + r"[a-z]:\\" + "|c:/" + "users" + "|/" + "users" + "/)")
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HTML_IMAGE_PATTERN = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)
COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_html_comments(text: str) -> str:
    return COMMENT_PATTERN.sub("", text)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_row(rows: list[dict[str, str]], column: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(column) == value:
            return row
    raise AssertionError(f"Could not find {column}={value!r}")


def github_anchor(heading: str) -> str:
    normalized = unicodedata.normalize("NFKD", heading.strip().lower())
    kept = []
    for char in normalized:
        if char.isalnum() or char in {" ", "-"}:
            kept.append(char)
    anchor = "".join(kept)
    anchor = re.sub(r"\s+", "-", anchor)
    anchor = re.sub(r"-+", "-", anchor)
    return anchor.strip("-")


def visible_headings(text: str) -> dict[str, str]:
    headings: dict[str, str] = {}
    for match in re.finditer(r"^(#{2,6})\s+(.+?)\s*$", strip_html_comments(text), re.MULTILINE):
        heading = match.group(2).strip()
        headings[heading] = github_anchor(heading)
    return headings


def section_text(text: str, heading: str) -> str:
    pattern = re.compile(rf"^##+\s+{re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return ""
    next_heading = re.search(r"^##\s+", text[match.end() :], re.MULTILINE)
    if not next_heading:
        return text[match.end() :]
    return text[match.end() : match.end() + next_heading.start()]


def relative_link_target(target: str) -> str:
    target = target.strip()
    if "#" in target:
        target = target.split("#", 1)[0]
    return target


def visible_markdown_links(text: str) -> list[str]:
    return [match.group(1) for match in LINK_PATTERN.finditer(strip_html_comments(text))]


def visible_markdown_images(text: str) -> list[str]:
    return [match.group(1) for match in IMAGE_PATTERN.finditer(strip_html_comments(text))]


def visible_local_images(text: str) -> list[str]:
    visible = strip_html_comments(text)
    return [match.group(1) for match in IMAGE_PATTERN.finditer(visible)] + [match.group(1) for match in HTML_IMAGE_PATTERN.finditer(visible)]


def assert_contains(errors: list[str], text: str, token: str, label: str) -> None:
    if token not in text:
        errors.append(f"Missing {label}: {token}")


def validate(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    readme = root / "README.md"

    if not readme.exists():
        return ["README.md is missing"]

    text = read_text(readme)
    visible = strip_html_comments(text)
    lower_visible = visible.lower()
    headings = visible_headings(text)

    for section in REQUIRED_SECTIONS:
        if section not in headings:
            errors.append(f"Missing README section: {section}")

    for directory in REQUIRED_ASSET_DIRS:
        if not (root / directory).is_dir():
            errors.append(f"Missing README asset directory: {directory}")

    if not IMAGE_PLAN.exists():
        errors.append("Missing docs/assets/readme/README_IMAGE_PLAN.md")
    else:
        plan_text = read_text(IMAGE_PLAN)
        for image_path in EXPECTED_IMAGE_PATHS:
            if image_path not in plan_text:
                errors.append(f"Image plan does not mention {image_path}")

    if PRIVATE_PATH_PATTERN.search(text):
        errors.append("README contains a private absolute filesystem path")

    for phrase in FORBIDDEN_PUBLIC_PHRASES:
        if phrase in lower_visible:
            errors.append(f"README contains forbidden phrase: {phrase}")

    for token, label in [
        (FINAL_SYSTEM, "final system name"),
        (FINAL_MODEL, "final selected predictive system name"),
        (BACKBONE, "physics-guided backbone name"),
        ("not a learned KAN component", "deployed system is not KAN wording"),
        ("KAN experiments were not selected for deployment", "KAN non-selection wording"),
        ("15 < base_rul <= 25", "frozen guard rule"),
        ("correction_bound: 10.0 cycles", "guard correction bound"),
        ("0.5 cycles", "guard margin"),
        ("global split conformal", "uncertainty method"),
        ("URGENT_ENGINEERING_REVIEW", "urgent maintenance action"),
        ("SCHEDULE_MAINTENANCE", "scheduled maintenance action"),
        ("PLAN_INSPECTION", "inspection maintenance action"),
        ("CONTINUE_MONITORING", "monitoring maintenance action"),
    ]:
        assert_contains(errors, text, token, label)

    headline_rows = csv_rows(root / HEADLINE_CSV.relative_to(ROOT))
    final_headline = first_row(headline_rows, "Status", "Final selected system")
    patch_headline = first_row(headline_rows, "Family", "Patch Transformer")

    fixed_rows = csv_rows(root / FIXED_POLICY_CSV.relative_to(ROOT))
    final_policy = first_row(fixed_rows, "model_id", "critical_boundary_safety_guarded_transformer")
    patch_policy = first_row(fixed_rows, "model_id", "patch_transformer_10x5_mean_b")

    registry_rows = csv_rows(root / MODEL_REGISTRY_CSV.relative_to(ROOT))

    required_metric_tokens = [
        f"{float(final_headline['Overall MAE']):.4f}",
        f"{float(final_headline['Overall RMSE']):.4f}",
        f"{float(final_headline['Severe optimism']):.4f}",
        f"{float(final_headline['Operational recall']):.4f}",
        f"{float(final_headline['Review workload']):.4f}",
        f"| Critical misses | {int(float(final_headline['Critical misses']))} |",
        f"| Model/system candidates evaluated | {len(registry_rows)} |",
    ]
    for token in required_metric_tokens:
        assert_contains(errors, text, token, "headline metric")

    mae_delta = float(final_headline["Overall MAE"]) - float(patch_headline["Overall MAE"])
    rmse_delta = float(final_headline["Overall RMSE"]) - float(patch_headline["Overall RMSE"])
    miss_delta = int(float(final_policy["critical_miss_count"])) - int(float(patch_policy["critical_miss_count"]))
    recall_delta = float(final_policy["operational_recall"]) - float(patch_policy["operational_recall"])
    workload_delta = float(final_policy["review_workload"]) - float(patch_policy["review_workload"])
    for token in [
        f"{mae_delta:+.4f} cycles",
        f"{rmse_delta:+.4f} cycles",
        f"{miss_delta:+d}",
        f"{recall_delta:+.4f}",
        f"{workload_delta:+.4f}",
    ]:
        assert_contains(errors, text, token, "patch-transformer improvement metric")

    anchors = set(headings.values())
    for target in visible_markdown_links(text):
        if target.startswith("#"):
            anchor = target[1:]
            if anchor not in anchors:
                errors.append(f"README link points to missing anchor: {target}")
            continue
        if re.match(r"^[a-z]+://", target) or target.startswith("mailto:"):
            continue
        rel = relative_link_target(target)
        if rel and not (root / rel).exists():
            errors.append(f"README link points to missing file: {target}")

    for target in visible_local_images(text):
        if re.match(r"^[a-z]+://", target):
            continue
        rel = relative_link_target(target)
        if rel and not (root / rel).exists():
            errors.append(f"Visible README image points to missing file: {target}")

    for image_path in EXPECTED_IMAGE_PATHS:
        if image_path not in text:
            errors.append(f"README missing commented placeholder for {image_path}")

    production_section = section_text(text, "Production Inference Output")
    if re.search(r"\b(true_rul|true RUL|ground[- ]truth|actual RUL)\b", production_section):
        errors.append("Production inference section exposes benchmark true-RUL labels")

    return errors


def main() -> int:
    errors = validate(ROOT)
    if errors:
        print("README validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("README validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
