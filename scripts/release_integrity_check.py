from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "release_readiness" / "release_integrity_check.json"

FORBIDDEN_DOCKER_FILES = [
    "Dockerfile",
    ".dockerignore",
    "docker-compose.yml",
]
WEAK_PUBLIC_LABELS = [
    "repository-derived " + "baseline",
    "repo " + "baseline",
    "downloaded " + "model",
    "bad " + "model",
    "failed " + "model",
    "Phase 5C " + "model",
    "Phase 5D " + "model",
]
TEXT_SUFFIXES = {".cff", ".csv", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def run(args: list[str]) -> tuple[int, str]:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    return result.returncode, result.stdout + result.stderr


def intended_files() -> list[Path]:
    code, output = run(["git", "ls-files", "--cached", "--others", "--exclude-standard"])
    if code != 0:
        return [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    return [ROOT / line.strip() for line in output.splitlines() if line.strip()]


def text_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore", ".gitattributes"}:
            yield path


def check_license() -> list[str]:
    failures: list[str] = []
    license_path = ROOT / "LICENSE"
    if not license_path.exists():
        failures.append("LICENSE is missing.")
        return failures
    text = license_path.read_text(encoding="utf-8")
    if "Apache License" not in text or "Version 2.0" not in text:
        failures.append("LICENSE does not contain Apache License 2.0 text.")
    if "Copyright 2026 Yarroju Rithvik" not in text:
        failures.append("LICENSE copyright holder is not the expected release owner.")
    if (ROOT / "LICENSE_REVIEW_REQUIRED.md").exists():
        failures.append("LICENSE_REVIEW_REQUIRED.md should be removed after license resolution.")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if 'license = "Apache-2.0"' not in pyproject:
        failures.append("pyproject.toml does not declare Apache-2.0.")
    return failures


def check_license_audit() -> list[str]:
    failures: list[str] = []
    path = ROOT / "reports" / "release_readiness" / "license_compatibility_audit.csv"
    if not path.exists():
        return ["license_compatibility_audit.csv is missing."]
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            status = row.get("status", "")
            compatible = row.get("apache_2_compatible", "")
            copied = row.get("copied_or_adapted", "")
            if status not in {"resolved", "excluded", "resolved_excluded"}:
                failures.append(f"Unresolved license audit row: {row.get('project_path')}")
            if copied == "Yes" and compatible != "Yes":
                failures.append(f"Retained copied/adapted row is not Apache-compatible: {row.get('project_path')}")
    return failures


def check_manifest() -> list[str]:
    failures: list[str] = []
    sys.path.insert(0, str(ROOT / "src"))
    from aeroguard.inference.artifact_loader import load_manifest, validate_component_hashes

    manifest = load_manifest(ROOT / "artifacts" / "final_release" / "frozen_system_manifest.json")
    if manifest.get("release_version") != "1.0.0":
        failures.append("Unexpected release version.")
    if manifest.get("model_version") != "aeroguard-phm-safety-v1":
        failures.append("Unexpected model version.")
    guard = manifest.get("safety_guard", {})
    expected_guard = {"boundary_low": 15.0, "boundary_high": 25.0, "margin": 0.5, "correction_bound": 10.0}
    for key, expected in expected_guard.items():
        if float(guard.get(key, -1)) != expected:
            failures.append(f"Frozen safety guard changed: {key}")
    hash_mismatches = validate_component_hashes(manifest)
    if hash_mismatches:
        failures.append(f"Manifest component hash mismatches: {hash_mismatches}")
    return failures


def check_text_surface(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    private_pattern = re.compile("(?i)(" + r"[a-z]:\\" + "|c:/" + "users" + "|/" + "users" + "/)")
    for path in text_files(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if private_pattern.search(text):
            failures.append(f"Private path pattern in {rel}")
        public_text = rel.startswith(("README.md", "MODEL_CARD.md", "REPRODUCIBILITY.md", "reports/final_release/")) or (
            rel.startswith("docs/") and not rel.startswith("docs/archive/")
        )
        if public_text:
            lower = text.lower()
            for phrase in WEAK_PUBLIC_LABELS:
                if phrase.lower() in lower:
                    failures.append(f"Weak public label in {rel}: {phrase}")
    return failures


def check_no_docker(paths: list[Path]) -> list[str]:
    failures = []
    names = {path.relative_to(ROOT).as_posix() for path in paths}
    for name in FORBIDDEN_DOCKER_FILES:
        if name in names or (ROOT / name).exists():
            failures.append(f"Docker artifact is intentionally deferred and should not exist: {name}")
    if any(".github/workflows" in path.relative_to(ROOT).as_posix() and "docker" in path.read_text(encoding="utf-8", errors="ignore").lower() for path in paths if path.suffix in {".yml", ".yaml"}):
        failures.append("GitHub Actions workflow appears to contain Docker-related content.")
    return failures


def check_release_audit() -> list[str]:
    code, output = run([sys.executable, "scripts/release_readiness_audit.py"])
    if code != 0:
        return [f"release_readiness_audit.py failed: {output.strip()}"]
    report = json.loads((ROOT / "reports" / "release_readiness" / "private_information_audit.json").read_text(encoding="utf-8"))
    failures = []
    if report.get("likely_secret_count"):
        failures.append("Likely secret candidates remain.")
    if report.get("public_documentation_private_path_hits"):
        failures.append("Private paths remain in public documentation.")
    manifest = json.loads((ROOT / "reports" / "release_readiness" / "public_release_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("release_blockers"):
        failures.append(f"Release blockers remain: {manifest['release_blockers']}")
    return failures


def main() -> int:
    paths = intended_files()
    failures: list[str] = []
    failures.extend(check_no_docker(paths))
    failures.extend(check_license())
    failures.extend(check_license_audit())
    failures.extend(check_manifest())
    failures.extend(check_text_surface(paths))
    failures.extend(check_release_audit())
    payload = {"status": "pass" if not failures else "fail", "failure_count": len(failures), "failures": failures}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
