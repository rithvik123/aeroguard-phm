"""Frozen AeroGuard-PHM predictor interface."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aeroguard.inference.artifact_loader import component_path, load_manifest, load_pickle
from aeroguard.inference.explanations import build_explanation
from aeroguard.inference.maintenance import recommend_maintenance
from aeroguard.inference.monitoring import build_inference_log
from aeroguard.inference.preprocessing import infer_engine_id, support_from_issues, validate_engine_history, validation_payload
from aeroguard.inference.safety_guard import apply_critical_boundary_guard
from aeroguard.inference.uncertainty import conformal_intervals


class AeroGuardPredictor:
    """Production-facing wrapper for the frozen final system.

    The wrapper never trains or tunes models. It attempts to load the frozen
    physics-guided Patch Transformer and preprocessor. If neural inference is
    unavailable for a local runtime reason, it falls back to a deterministic
    manifest compatibility path for smoke tests and demos; that fallback is
    explicitly reported in output warnings.
    """

    def __init__(self, manifest: dict[str, Any], *, device: str = "cpu") -> None:
        self.manifest = manifest
        self.device = device
        self.model_version = str(manifest.get("model_version", "aeroguard-phm-safety-v1"))
        self._preprocessor: dict[str, Any] | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._runtime_warning: str | None = None
        self._load_runtime()

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, *, device: str = "cpu") -> "AeroGuardPredictor":
        return cls(load_manifest(manifest_path), device=device)

    def predict_engine(self, engine_history: pd.DataFrame) -> dict[str, Any]:
        start = time.perf_counter()
        frame = engine_history.copy()
        issues = validate_engine_history(
            frame,
            min_history=int(self.manifest.get("minimum_history_length", 10)),
            max_history=int(self.manifest.get("maximum_history_length", 500)),
        )
        validation = validation_payload(issues)
        engine_id = infer_engine_id(frame)
        support_status, support_score = support_from_issues(issues)
        if not validation["valid"]:
            return {
                "engine_id": engine_id,
                "model_version": self.model_version,
                "valid": False,
                "errors": validation["errors"],
                "warnings": validation["warnings"],
            }

        base_rul, operating_regime, model_warnings = self._predict_base_rul(frame)
        max_rul = float(self.manifest.get("maximum_output_rul", 250.0))
        if base_rul > max_rul:
            model_warnings.append(f"Base RUL exceeded plausible output range and was clipped to {max_rul:g} cycles.")
            base_rul = max_rul
        guard_config = self.manifest.get("safety_guard", {})
        guarded = apply_critical_boundary_guard(
            base_rul,
            boundary_low=float(guard_config.get("boundary_low", 15.0)),
            boundary_high=float(guard_config.get("boundary_high", 25.0)),
            margin=float(guard_config.get("margin", 0.5)),
            bound=float(guard_config.get("correction_bound", 10.0)),
        )
        uncertainty = conformal_intervals(float(guarded["safety_adjusted_rul"]), self.manifest.get("uncertainty", {}))
        maintenance = recommend_maintenance(float(guarded["safety_adjusted_rul"]), self.manifest.get("maintenance_policy", {}))
        warning_payload = validation["warnings"] + [{"code": "runtime_warning", "message": item, "severity": "warning", "column": None} for item in model_warnings]
        interval_width_90 = uncertainty.get("interval_width_90")
        result: dict[str, Any] = {
            "engine_id": engine_id,
            "model_version": self.model_version,
            "valid": True,
            **guarded,
            **uncertainty,
            "operating_regime": operating_regime,
            "support_status": support_status,
            "support_score": support_score,
            **maintenance,
            "warnings": warning_payload,
        }
        result["explanation"] = build_explanation(
            guard_active=bool(result["safety_guard_activated"]),
            maintenance_action=str(result["maintenance_action"]),
            interval_width_90=interval_width_90,
            support_status=support_status,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        result["inference_latency_ms"] = latency_ms
        result["monitoring_log"] = build_inference_log(
            result,
            input_columns=list(frame.columns),
            input_row_count=len(frame),
            latency_ms=latency_ms,
        )
        return result

    def predict_batch(self, engine_histories: list[pd.DataFrame]) -> list[dict[str, Any]]:
        return [self.predict_engine(history) for history in engine_histories]

    def validate_engine(self, engine_history: pd.DataFrame) -> dict[str, Any]:
        issues = validate_engine_history(engine_history)
        return validation_payload(issues)

    def _load_runtime(self) -> None:
        try:
            import torch

            from aeroguard.deep.models.physics_guided_patch_transformer import PhysicsGuidedPatchTransformer

            self._torch = torch
            preprocessor_path = component_path(self.manifest, "final_preprocessor")
            self._preprocessor = load_pickle(preprocessor_path)
            feature_count = len(self._preprocessor["features"])
            architecture = self.manifest.get("backbone_architecture", {})
            model = PhysicsGuidedPatchTransformer(
                input_dim=feature_count + 1,
                window_length=int(architecture.get("window_length", 50)),
                patch_length=int(architecture.get("patch_length", 10)),
                patch_stride=int(architecture.get("patch_stride", 5)),
                projection_dim=int(architecture.get("projection_dim", 64)),
                layers=int(architecture.get("layers", 2)),
                heads=int(architecture.get("heads", 4)),
                feedforward_dim=int(architecture.get("feedforward_dim", 192)),
                dropout=float(architecture.get("dropout", 0.15)),
                positional_encoding=str(architecture.get("positional_encoding", "learnable")),
                pooling=str(architecture.get("pooling", "mean")),
                causal_attention=bool(architecture.get("causal_attention", False)),
                health_head_enabled=True,
                rate_head_enabled=True,
                output_activation="softplus",
                validate_inputs=True,
            )
            checkpoint_path = component_path(self.manifest, "physics_regime_checkpoint")
            payload = torch.load(checkpoint_path, map_location=self.device)
            state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            model.load_state_dict(state, strict=False)
            model.to(self.device)
            model.eval()
            self._model = model
        except Exception as exc:  # pragma: no cover - depends on local optional runtime state
            self._runtime_warning = f"Neural checkpoint runtime unavailable; using deterministic compatibility predictor ({type(exc).__name__}: {exc})."
            self._model = None

    def _predict_base_rul(self, frame: pd.DataFrame) -> tuple[float, int | None, list[str]]:
        warnings: list[str] = []
        if self._model is not None and self._preprocessor is not None and self._torch is not None:
            try:
                value, regime = self._predict_with_checkpoint(frame)
                return value, regime, warnings
            except Exception as exc:
                warnings.append(f"Checkpoint inference failed; used deterministic compatibility predictor ({type(exc).__name__}: {exc}).")
        elif self._runtime_warning:
            warnings.append(self._runtime_warning)
        value = self._deterministic_base_rul(frame)
        regime = self._latest_regime(frame)
        return value, regime, warnings

    def _predict_with_checkpoint(self, frame: pd.DataFrame) -> tuple[float, int | None]:
        assert self._preprocessor is not None
        assert self._torch is not None
        normalizer = self._preprocessor["normalizer"]
        scaler = self._preprocessor["scaler"]
        features = self._preprocessor["features"]
        transformed = normalizer.transform(frame.copy())
        transformed.loc[:, features] = scaler.transform(transformed[features])
        values = transformed[features].to_numpy(dtype=np.float32)
        window_length = int(self.manifest.get("window_length", 50))
        tail = values[-window_length:]
        mask = np.ones((len(tail), 1), dtype=np.float32)
        if len(tail) < window_length:
            pad = np.zeros((window_length - len(tail), values.shape[1]), dtype=np.float32)
            pad_mask = np.zeros((window_length - len(tail), 1), dtype=np.float32)
            tail = np.vstack([pad, tail])
            mask = np.vstack([pad_mask, mask])
        sequence = np.concatenate([tail, mask], axis=1)[None, :, :]
        tensor = self._torch.as_tensor(sequence, dtype=self._torch.float32, device=self.device)
        with self._torch.no_grad():
            output = self._model(tensor)
            prediction = output["rul_prediction"]
        value = float(prediction.detach().cpu().numpy().reshape(-1)[0])
        regime = int(transformed["operating_regime"].iloc[-1]) if "operating_regime" in transformed.columns else None
        return max(0.0, value), regime

    def _deterministic_base_rul(self, frame: pd.DataFrame) -> float:
        cycle = float(pd.to_numeric(frame["cycle"], errors="coerce").max())
        sensor_2 = pd.to_numeric(frame.get("sensor_2", pd.Series([0.0])), errors="coerce").fillna(0.0)
        sensor_15 = pd.to_numeric(frame.get("sensor_15", pd.Series([0.0])), errors="coerce").fillna(0.0)
        trend = 0.0
        if len(sensor_2) >= 2:
            trend += float(sensor_2.iloc[-1] - sensor_2.iloc[0]) * 0.8
        if len(sensor_15) >= 2:
            trend += float(sensor_15.iloc[-1] - sensor_15.iloc[0]) * 40.0
        return max(0.0, min(250.0, 125.0 - 0.9 * cycle - trend))

    def _latest_regime(self, frame: pd.DataFrame) -> int | None:
        if self._preprocessor is not None:
            try:
                transformed = self._preprocessor["normalizer"].transform(frame.copy())
                if "operating_regime" in transformed.columns:
                    return int(transformed["operating_regime"].iloc[-1])
            except Exception:
                return None
        return None
