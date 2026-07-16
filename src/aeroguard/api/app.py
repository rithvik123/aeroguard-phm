"""FastAPI service for the frozen AeroGuard-PHM final release."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

try:  # pragma: no cover - exercised by import tests when available
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - optional dependency guard
    FastAPI = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]
    Field = lambda default=None, **_: default  # type: ignore[assignment]

from aeroguard.inference.artifact_loader import validate_component_hashes
from aeroguard.inference.predictor import AeroGuardPredictor
from aeroguard.inference.schemas import REQUIRED_INPUT_COLUMNS


DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "artifacts" / "final_release" / "frozen_system_manifest.json"
_PREDICTOR: AeroGuardPredictor | None = None
_LOAD_ERROR: str | None = None


if BaseModel is not object:

    class EngineHistoryRequest(BaseModel):
        engine_id: str | None = None
        records: list[dict[str, Any]] = Field(default_factory=list)


    class BatchHistoryRequest(BaseModel):
        engines: list[EngineHistoryRequest] = Field(default_factory=list)

else:
    EngineHistoryRequest = dict  # type: ignore[misc,assignment]
    BatchHistoryRequest = dict  # type: ignore[misc,assignment]


def manifest_path() -> Path:
    return Path(os.environ.get("AEROGUARD_MANIFEST", str(DEFAULT_MANIFEST)))


def get_predictor() -> AeroGuardPredictor:
    global _PREDICTOR, _LOAD_ERROR
    if _PREDICTOR is not None:
        return _PREDICTOR
    try:
        _PREDICTOR = AeroGuardPredictor.from_manifest(manifest_path())
        _LOAD_ERROR = None
        return _PREDICTOR
    except Exception as exc:  # pragma: no cover - depends on generated manifest
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        raise


def _request_to_frame(request: EngineHistoryRequest) -> pd.DataFrame:
    payload = request.model_dump() if hasattr(request, "model_dump") else dict(request)
    records = payload.get("records", [])
    if not records:
        raise ValueError("records must not be empty")
    frame = pd.DataFrame(records)
    engine_id = payload.get("engine_id")
    if engine_id and "engine_id" not in frame.columns:
        frame.insert(0, "engine_id", engine_id)
    return frame


if FastAPI is not None:
    app = FastAPI(title="AeroGuard-PHM Safety-Guarded RUL System", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        path = manifest_path()
        return {
            "status": "ok" if path.exists() else "manifest_missing",
            "model_loaded": _PREDICTOR is not None,
            "manifest_available": path.exists(),
            "load_error": _LOAD_ERROR,
        }

    @app.get("/model")
    def model() -> dict[str, Any]:
        predictor = get_predictor()
        manifest = predictor.manifest
        return {
            "system_name": manifest.get("system_name"),
            "model_version": manifest.get("model_version"),
            "predictive_backbone": manifest.get("predictive_backbone"),
            "safety_layer": manifest.get("safety_layer"),
            "required_columns": REQUIRED_INPUT_COLUMNS,
            "hash_mismatches": len(validate_component_hashes(manifest)),
        }

    @app.post("/validate-input")
    def validate_input(request: EngineHistoryRequest) -> dict[str, Any]:
        try:
            frame = _request_to_frame(request)
            return get_predictor().validate_engine(frame)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/predict")
    def predict(request: EngineHistoryRequest) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            frame = _request_to_frame(request)
            result = get_predictor().predict_engine(frame)
            result["api_latency_ms"] = (time.perf_counter() - started) * 1000.0
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/predict-batch")
    def predict_batch(request: BatchHistoryRequest) -> dict[str, Any]:
        started = time.perf_counter()
        payload = request.model_dump() if hasattr(request, "model_dump") else dict(request)
        try:
            frames = [_request_to_frame(EngineHistoryRequest(**item)) for item in payload.get("engines", [])]
            predictions = get_predictor().predict_batch(frames)
            return {
                "model_version": get_predictor().model_version,
                "count": len(predictions),
                "predictions": predictions,
                "api_latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

else:
    app = None
