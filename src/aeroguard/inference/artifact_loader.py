"""Manifest and artifact loading utilities for the frozen final release."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_path(path_value: str | Path, manifest_dir: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    base = project_root() if manifest_dir is None else manifest_dir
    return (base / path).resolve()


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    manifest["_manifest_path"] = str(Path(manifest_path).resolve())
    return manifest


def component_path(manifest: dict[str, Any], component_name: str) -> Path:
    components = manifest.get("components", {})
    component = components.get(component_name)
    if not isinstance(component, dict) or "path" not in component:
        raise KeyError(f"Manifest component not found: {component_name}")
    return resolve_manifest_path(component["path"])


def validate_component_hashes(manifest: dict[str, Any]) -> list[dict[str, str]]:
    """Return hash mismatch records; an empty list means all present hashes match."""

    mismatches: list[dict[str, str]] = []
    for name, component in manifest.get("components", {}).items():
        if not isinstance(component, dict):
            continue
        path_text = component.get("path")
        expected = component.get("sha256")
        if not path_text or not expected:
            continue
        path = resolve_manifest_path(path_text)
        if not path.exists():
            mismatches.append({"component": name, "path": str(path), "reason": "missing"})
            continue
        actual = sha256_file(path)
        if actual != expected:
            mismatches.append(
                {
                    "component": name,
                    "path": str(path),
                    "expected_sha256": str(expected),
                    "actual_sha256": actual,
                }
            )
    return mismatches
