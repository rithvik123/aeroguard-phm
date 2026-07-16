from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "release_readiness"
LARGE_THRESHOLD = 10 * 1024 * 1024
HUGE_THRESHOLD = 50 * 1024 * 1024

TEXT_EXTS = {
    ".bat",
    ".cfg",
    ".cff",
    ".csv",
    ".env",
    ".example",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

PUBLIC_DOC_EXTS = {".md", ".rst", ".txt", ".cff"}
PRIVATE_PATTERNS = {
    "windows_absolute_path": re.compile("(?i)" + r"\b[A-Z]" + ":" + r"\\"),
    "user_profile_path": re.compile("(?i)" + "C:" + r"\\Users\\" + r"[^\\\s]+"),
    "conda_user_env": re.compile(r"(?i)\.conda\\envs"),
    "unix_user_path": re.compile("(?i)" + "/" + "Users" + "/" + r"[^/\s]+"),
    "local_temp_path": re.compile("(?i)" + r"(\\AppData\\Local\\Temp|" + "/" + "tmp" + "/" + ")"),
}
CURRENT_USERNAME = os.environ.get("USERNAME") or os.environ.get("USER")
if CURRENT_USERNAME:
    PRIVATE_PATTERNS["current_username"] = re.compile(rf"(?i)\b{re.escape(CURRENT_USERNAME)}\b")
SECRET_PATTERNS = {
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "bearer_token": re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    "password_assignment": re.compile(r"(?i)\b(password|passwd|pwd)\s*[:=]\s*['\"]?[^'\"\s]{6,}"),
    "api_key_assignment": re.compile(r"(?i)\b(api[_-]?key|secret|client_secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9._~+/=-]{16,}"),
    "database_url": re.compile(r"(?i)\b(postgres|mysql|mongodb|redis)://[^\\s]+"),
    "jwt": re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
}
WEAK_PUBLIC_LABEL_TERMS = [
    "repository-derived " + "baseline",
    "repo " + "baseline",
    "downloaded " + "repo",
    "original " + "repo model",
    "Phase 5C " + "model",
    "Phase 5D " + "model",
    "bad " + "model",
    "failed " + "model",
]
UPSTREAM_PATTERNS = {
    "production_pdm_system": re.compile(r"production-pdm-system|Predictive Maintenance Manufacturing System", re.I),
    "original_aircraft_pm": re.compile(r"original-aircraft-pm|Predictive Maintenance \(PdM\) of Aircraft Engine", re.I),
    "predictive_maintenance_generic": re.compile(r"\bpredictive-maintenance\b", re.I),
    "weak_public_label": re.compile("|".join(re.escape(term) for term in WEAK_PUBLIC_LABEL_TERMS), re.I),
    "old_dataset_copy": re.compile(r"WA_Fn-UseC|Telco|ford\.csv", re.I),
}


@dataclass
class FileRecord:
    path: Path
    rel: str
    size: int
    suffix: str


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_files() -> Iterable[FileRecord]:
    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        try:
            rel_path = rel(path)
        except ValueError:
            continue
        if rel_path.startswith(".git/"):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        yield FileRecord(path=path, rel=rel_path, size=size, suffix=path.suffix.lower())


def is_text(record: FileRecord) -> bool:
    name = record.path.name.lower()
    return record.suffix in TEXT_EXTS or name in {".gitignore", ".gitattributes", ".env.example"}


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8-sig")
        except Exception:
            return None
    except Exception:
        return None


def classify(record: FileRecord) -> tuple[str, str, str, bool, str]:
    rel_path = record.rel
    parts = rel_path.split("/")
    path_lower = rel_path.lower()

    if "__pycache__" in parts or ".pytest_cache" in parts or path_lower.endswith((".pyc", ".pyo")):
        return ("Remove", "Remove", "Generated Python/test cache.", False, "Removed or ignored")
    if rel_path.startswith(("references/", "extracted-code/", "notebooks/original-aircraft-pm/", "data/reference-derived/")):
        return ("Remove", "Remove", "Copied upstream/reference repository material not imported by AeroGuard-PHM.", True, "Removed or ignored")
    if rel_path.startswith("data/raw/"):
        return ("Add to .gitignore", "Ignore", "Raw C-MAPSS dataset should not be staged by default.", True, "Ignored")
    if rel_path.startswith("outputs/") or rel_path.startswith("notes/"):
        return ("Remove", "Remove", "One-off local audit/scratch material.", False, "Removed or ignored")
    if rel_path.startswith("reports/release_readiness/"):
        return ("Keep", "Keep", "Public release-readiness report.", False, "Kept")
    if rel_path.startswith("reports/final_release/"):
        return ("Keep", "Keep", "Curated final-release evidence.", False, "Kept")
    if rel_path.startswith("reports/aerokan_phm_critical_gate/"):
        keep_names = {
            "benchmark_predictions.csv",
            "benchmark_metrics.json",
            "benchmark_fixed_policy_metrics.json",
            "metric_definition_audit.json",
            "locked_maintenance_policy.json",
            "locked_uncertainty_method.json",
            "cascade_metadata.json",
        }
        if record.path.name in keep_names:
            return ("Keep", "Keep", "Curated final selected-system evidence.", False, "Kept")
        return ("Add to .gitignore", "Ignore", "Detailed generated experiment output.", False, "Ignored")
    if rel_path.startswith("reports/"):
        return ("Add to .gitignore", "Ignore", "Generated intermediate report output.", False, "Ignored")
    if rel_path.startswith("artifacts/final_release/"):
        return ("Keep", "Keep", "Release manifest artifact.", False, "Kept")
    if rel_path.startswith("artifacts/physics_guided_rul/checkpoints/") and record.path.name in {"locked_physics_guided_model.pt", "final_preprocessor.pkl"}:
        return ("Keep", "Keep", "Small frozen inference artifact required by final manifest.", False, "Kept")
    if rel_path.startswith("artifacts/aerokan_phm_critical_gate/") and record.path.name in {"cascade_metadata.json", "uncertainty_model.pkl", "maintenance_policy.pkl"}:
        return ("Keep", "Keep", "Small frozen guard/uncertainty/policy artifact required by final manifest.", False, "Kept")
    if rel_path.startswith("artifacts/"):
        return ("Add to .gitignore", "Ignore", "Generated model artifact not required for public inference.", False, "Ignored")
    if rel_path.startswith("docs/assets/readme/"):
        return ("Keep", "Keep", "README asset or asset plan.", False, "Kept")
    if rel_path in {"LICENSE", "THIRD_PARTY_NOTICES.md", "CITATION.cff", "CONTRIBUTING.md", "SECURITY.md", "CODE_OF_CONDUCT.md", ".gitignore", ".gitattributes", ".env.example", "pyproject.toml"}:
        return ("Keep", "Keep", "Public release metadata.", False, "Kept")
    return ("Keep", "Keep", "Project source, config, docs, tests or curated example.", False, "Kept")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mask(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return value[:4] + "***" + value[-4:]


def cleanup_inventory(records: list[FileRecord]) -> list[dict[str, object]]:
    rows = []
    for record in records:
        category, action, reason, attribution, final_action = classify(record)
        rows.append(
            {
                "Path": record.rel,
                "File type": record.suffix or record.path.name,
                "File size": record.size,
                "Category": category,
                "Proposed action": action,
                "Reason": reason,
                "Attribution required": "Yes" if attribution else "No",
                "Safe to delete": "Yes" if category == "Remove" and not attribution else "No",
                "Final action": final_action,
            }
        )
    return rows


def upstream_audit(records: list[FileRecord]) -> list[dict[str, object]]:
    rows = []
    for record in records:
        if record.rel.startswith("reports/release_readiness/"):
            continue
        if not is_text(record):
            continue
        text = read_text(record.path)
        if text is None:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for name, pattern in UPSTREAM_PATTERNS.items():
                if pattern.search(line):
                    if name == "weak_public_label" and record.rel.startswith(("scripts/", "tests/")):
                        action = "Retained as validator/test forbidden phrase."
                    elif record.rel.startswith(("references/", "extracted-code/", "notebooks/original-aircraft-pm/", "data/reference-derived/")):
                        action = "Excluded from public staging and cleanup target."
                    else:
                        action = "Reviewed; keep only when legitimate AeroGuard terminology or citation."
                    rows.append(
                        {
                            "File": record.rel,
                            "Line or location": line_no,
                            "Detected reference": line.strip()[:240],
                            "Category": name,
                            "Action taken": action,
                            "Attribution impact": "See THIRD_PARTY_NOTICES.md" if name in {"production_pdm_system", "original_aircraft_pm"} else "None",
                        }
                    )
    return rows


def private_and_secret_audits(records: list[FileRecord]) -> tuple[dict[str, object], list[dict[str, object]]]:
    private_hits = []
    secret_hits = []
    for record in records:
        if record.rel.startswith("reports/release_readiness/"):
            continue
        if not is_text(record):
            continue
        text = read_text(record.path)
        if text is None:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for name, pattern in PRIVATE_PATTERNS.items():
                for match in pattern.finditer(line):
                    private_hits.append(
                        {
                            "file": record.rel,
                            "line": line_no,
                            "kind": name,
                            "match": mask(match.group(0)),
                            "public_release_action": "Historical frozen metadata tolerated only if file is ignored or documented; public docs/source should be sanitized.",
                        }
                    )
            for name, pattern in SECRET_PATTERNS.items():
                for match in pattern.finditer(line):
                    candidate = match.group(0)
                    if record.path.name == ".env.example" or "example" in record.rel.lower():
                        likelihood = "example_placeholder"
                    elif "minioadmin" in candidate.lower() or "admin/admin" in line.lower():
                        likelihood = "default_demo_credential"
                    else:
                        likelihood = "review_required"
                    secret_hits.append(
                        {
                            "file": record.rel,
                            "line": line_no,
                            "kind": name,
                            "masked_candidate": mask(candidate),
                            "likelihood": likelihood,
                        }
                    )
    public_doc_hits = [hit for hit in private_hits if Path(hit["file"]).suffix.lower() in PUBLIC_DOC_EXTS and not str(hit["file"]).startswith("docs/archive/")]
    likely_secrets = [hit for hit in secret_hits if hit["likelihood"] == "review_required"]
    private_report = {
        "total_private_path_hits": len(private_hits),
        "public_documentation_private_path_hits": public_doc_hits,
        "private_path_hits": private_hits[:500],
        "likely_secret_count": len(likely_secrets),
        "secret_scan_result": "fail" if likely_secrets else "pass",
        "secret_hits": secret_hits[:500],
    }
    return private_report, secret_hits


def large_files(records: list[FileRecord]) -> list[dict[str, object]]:
    rows = []
    for record in sorted(records, key=lambda item: item.size, reverse=True):
        if record.size < LARGE_THRESHOLD:
            continue
        category, action, reason, _, _ = classify(record)
        if record.rel.startswith("data/raw/"):
            recommendation = "Ignore raw dataset; document download/source separately."
            alt = "Download C-MAPSS from the authoritative source."
        elif record.rel.startswith("artifacts/") and action == "Keep":
            recommendation = "Commit only if required for frozen inference; otherwise use release storage."
            alt = "Use model release assets."
        elif action == "Keep":
            recommendation = "Review before committing; large but classified as public evidence."
            alt = "Compress, summarize, or release externally if too large."
        else:
            recommendation = "Ignore or remove from public staging."
            alt = "Regenerate from scripts or keep locally."
        rows.append(
            {
                "Path": record.rel,
                "Size": record.size,
                "Type": record.suffix or record.path.name,
                "Commit recommendation": recommendation,
                "Reproduction alternative": alt,
            }
        )
    return rows


def run_git(args: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return 127, "git not found"
    return result.returncode, (result.stdout + result.stderr)


def git_ignored_paths(records: list[FileRecord]) -> set[str]:
    ignored_paths = set()
    if (ROOT / ".git").exists():
        for record in records:
            code, _ = run_git(["check-ignore", "-q", record.rel])
            if code == 0:
                ignored_paths.add(record.rel)
    return ignored_paths


def repository_size_report(records: list[FileRecord], ignored_paths: set[str]) -> dict[str, object]:
    largest = sorted(records, key=lambda item: item.size, reverse=True)[:30]
    def total(prefix: str) -> int:
        return sum(record.size for record in records if record.rel.startswith(prefix))

    return {
        "total_files": len(records),
        "total_working_tree_size_bytes": sum(record.size for record in records),
        "estimated_staged_size_bytes": sum(record.size for record in records if record.rel not in ignored_paths),
        "largest_30_files": [{"path": record.rel, "size": record.size} for record in largest],
        "files_above_10_mb": [{"path": record.rel, "size": record.size} for record in records if record.size >= LARGE_THRESHOLD],
        "files_above_50_mb": [{"path": record.rel, "size": record.size} for record in records if record.size >= HUGE_THRESHOLD],
        "ignored_data_size_bytes": sum(record.size for record in records if record.rel.startswith("data/") and record.rel in ignored_paths),
        "ignored_artifact_size_bytes": sum(record.size for record in records if record.rel.startswith("artifacts/") and record.rel in ignored_paths),
        "image_size_bytes": total("docs/assets/readme/"),
        "documentation_size_bytes": total("docs/") + sum(record.size for record in records if record.rel.endswith(".md") and "/" not in record.rel),
        "source_size_bytes": total("src/"),
    }


def public_manifest(records: list[FileRecord], private_report: dict[str, object], upstream_rows: list[dict[str, object]], large_rows: list[dict[str, object]], ignored_paths: set[str]) -> dict[str, object]:
    if (ROOT / ".git").exists():
        ignored = sorted(ignored_paths)
        intended = [record.rel for record in records if record.rel not in ignored_paths]
    else:
        ignored = []
        intended = []
        for record in records:
            category, *_ = classify(record)
            (ignored if category in {"Remove", "Add to .gitignore"} else intended).append(record.rel)

    image_paths = [
        "docs/assets/readme/architecture/model_development_journey.png",
        "docs/assets/readme/architecture/critical_boundary_safety_guard.png",
        "docs/assets/readme/hero/aeroguard_phm_hero.png",
        "docs/assets/readme/hero/rul_problem_statement.png",
        "docs/assets/readme/architecture/final_system_design.png",
    ]
    blockers = []
    if not (ROOT / "LICENSE").exists():
        blockers.append("Project license not selected; see LICENSE_REVIEW_REQUIRED.md.")
    if (ROOT / "LICENSE_REVIEW_REQUIRED.md").exists():
        blockers.append("License review blocker file still exists after license resolution.")
    if not (ROOT / "reports/release_readiness/license_compatibility_audit.csv").exists():
        blockers.append("License compatibility audit is missing.")
    if private_report.get("public_documentation_private_path_hits"):
        blockers.append("Private path remains in public documentation.")
    if private_report.get("likely_secret_count"):
        blockers.append("Likely secret candidate detected.")
    if any(row["Path"].startswith("data/raw/") and row["Path"] in intended for row in cleanup_inventory(records)):
        blockers.append("Raw dataset may be staged.")

    return {
        "project_name": "AeroGuard-PHM",
        "release_version": "1.0.0",
        "public_model_name": "Critical-Boundary Safety-Guarded Physics-Guided Transformer",
        "public_system_name": "AeroGuard-PHM Safety-Guarded RUL System",
        "files_intended_for_commit": sorted(intended),
        "files_intentionally_ignored": sorted(ignored),
        "large_file_exceptions": [row for row in large_rows if "required for frozen inference" in row.get("Commit recommendation", "")],
        "image_paths": image_paths,
        "documentation_files": sorted([record.rel for record in records if record.rel.endswith(".md") or record.rel.endswith(".cff")]),
        "test_result": "pending",
        "secret_scan_result": private_report.get("secret_scan_result"),
        "attribution_status": "THIRD_PARTY_NOTICES.md created; unresolved copied notebook license excluded from staging.",
        "license_status": "missing_project_license" if not (ROOT / "LICENSE").exists() else "project_license_present",
        "dataset_inclusion_status": "raw_data_ignored",
        "local_path_audit_result": "pass" if not private_report.get("public_documentation_private_path_hits") else "fail",
        "upstream_reference_audit_result": "see upstream_reference_audit.csv",
        "readme_validation_result": "pending",
        "release_blockers": blockers,
        "release_ready": not blockers,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = list(iter_files())
    ignored_paths = git_ignored_paths(records)
    cleanup_rows = cleanup_inventory(records)
    upstream_rows = upstream_audit(records)
    private_report, secret_hits = private_and_secret_audits([record for record in records if record.rel not in ignored_paths])
    large_rows = large_files(records)
    size_report = repository_size_report(records, ignored_paths)
    manifest = public_manifest(records, private_report, upstream_rows, large_rows, ignored_paths)

    write_csv(
        OUT / "cleanup_inventory.csv",
        cleanup_rows,
        ["Path", "File type", "File size", "Category", "Proposed action", "Reason", "Attribution required", "Safe to delete", "Final action"],
    )
    write_csv(
        OUT / "upstream_reference_audit.csv",
        upstream_rows,
        ["File", "Line or location", "Detected reference", "Category", "Action taken", "Attribution impact"],
    )
    write_csv(OUT / "large_files.csv", large_rows, ["Path", "Size", "Type", "Commit recommendation", "Reproduction alternative"])
    (OUT / "private_information_audit.json").write_text(json.dumps(private_report, indent=2), encoding="utf-8")
    (OUT / "repository_size_report.json").write_text(json.dumps(size_report, indent=2), encoding="utf-8")
    (OUT / "public_release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(records), "private_hits": private_report["total_private_path_hits"], "likely_secrets": private_report["likely_secret_count"], "large_files": len(large_rows), "blockers": manifest["release_blockers"]}, indent=2))
    return 1 if private_report["likely_secret_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
