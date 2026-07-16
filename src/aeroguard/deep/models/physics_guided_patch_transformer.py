"""Physics-guided Patch Transformer with auxiliary degradation heads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from aeroguard.deep.models.common import validate_parameter_budget
from aeroguard.deep.models.patch_transformer import PatchTemporalTransformerRegressor


class PhysicsGuidedPatchTransformer(PatchTemporalTransformerRegressor):
    """Patch Transformer encoder plus RUL, health-proxy, and degradation-rate heads."""

    def __init__(
        self,
        input_dim: int,
        window_length: int = 50,
        patch_length: int = 10,
        patch_stride: int = 5,
        projection_dim: int = 64,
        layers: int = 2,
        heads: int = 4,
        feedforward_dim: int = 192,
        dropout: float = 0.1,
        positional_encoding: str = "learnable",
        pooling: str = "mean",
        causal_attention: bool = False,
        health_head_enabled: bool = True,
        rate_head_enabled: bool = True,
        output_activation: str = "softplus",
        validate_inputs: bool = True,
        parameter_budget: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            window_length=window_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            projection_dim=projection_dim,
            layers=layers,
            heads=heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            positional_encoding=positional_encoding,
            pooling=pooling,
            causal_attention=causal_attention,
            **kwargs,
        )
        if output_activation not in {"softplus", "relu"}:
            raise ValueError("output_activation must be 'softplus' or 'relu'.")
        self.output_activation = output_activation
        self.validate_inputs = bool(validate_inputs)
        self.health_head_enabled = bool(health_head_enabled)
        self.rate_head_enabled = bool(rate_head_enabled)
        self.head = _ScalarHead(projection_dim, projection_dim, dropout, activation=None)
        self.health_head = _ScalarHead(projection_dim, projection_dim, dropout, activation="sigmoid") if self.health_head_enabled else None
        self.degradation_rate_head = _ScalarHead(projection_dim, projection_dim, dropout, activation="softplus") if self.rate_head_enabled else None
        self.model_kind = "physics_guided_patch_transformer"
        if parameter_budget is not None:
            validate_parameter_budget(self, int(parameter_budget))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        self._validate_input(x)
        encoded, patch_mask, _ = self.encode_tokens(x)
        latent = self.pool(encoded, patch_mask)
        raw = self.head(latent)
        prediction = F.softplus(raw) if self.output_activation == "softplus" else torch.relu(raw)
        health = None if self.health_head is None else self.health_head(latent)
        rate = None if self.degradation_rate_head is None else self.degradation_rate_head(latent)
        return {
            "rul_raw": raw,
            "rul_prediction": prediction,
            "health_score": health,
            "degradation_rate": rate,
            "latent": latent,
            "valid_token_count": patch_mask.sum(dim=1).to(dtype=torch.float32).view(-1, 1),
        }

    def warm_start_from_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        load_encoder_only: bool = True,
        strict: bool = False,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load compatible Phase 5B weights without modifying the checkpoint."""

        payload = torch.load(Path(checkpoint_path), map_location=map_location)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise ValueError("Checkpoint does not contain a state dictionary.")
        current = self.state_dict()
        if load_encoder_only:
            excluded = ("head.", "health_head.", "degradation_rate_head.")
            filtered = {
                key: value
                for key, value in state.items()
                if key in current and not key.startswith(excluded) and tuple(value.shape) == tuple(current[key].shape)
            }
        else:
            filtered = {key: value for key, value in state.items() if key in current and tuple(value.shape) == tuple(current[key].shape)}
        incompatible = self.load_state_dict(filtered, strict=False)
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys or len(filtered) != len(state)):
            raise RuntimeError("Strict warm start failed due to missing or unexpected keys.")
        return {
            "checkpoint_path": str(checkpoint_path),
            "load_encoder_only": bool(load_encoder_only),
            "loaded_key_count": int(len(filtered)),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }

    def _validate_input(self, x: torch.Tensor) -> None:
        if not self.validate_inputs:
            return
        if x.ndim != 3:
            raise ValueError("Expected input shape [batch, window_length, input_dim].")
        if x.shape[1] != self.window_length:
            raise ValueError("Incorrect window length.")
        if x.shape[2] != self.input_dim:
            raise ValueError("Incorrect feature count.")
        if not torch.isfinite(x).all():
            raise ValueError("Model input must be finite.")
        mask = x[..., -1]
        if ((mask < 0.0) | (mask > 1.0)).any():
            raise ValueError("The final input channel must be a [0, 1] validity mask.")


class _ScalarHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, activation: str | None) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.net(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(value)
        if self.activation == "softplus":
            return F.softplus(value)
        return value
