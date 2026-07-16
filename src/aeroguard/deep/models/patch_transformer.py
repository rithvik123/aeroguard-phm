"""Patch-based temporal Transformer for compact RUL sequence modelling."""

from __future__ import annotations

import math
import inspect

import torch
from torch import nn

from aeroguard.deep.models.attention_pooling import build_pooling, validate_token_mask
from aeroguard.deep.models.common import PositiveHead, split_features_mask
from aeroguard.deep.models.positional_encoding import build_positional_encoding


class PatchTemporalTransformerRegressor(nn.Module):
    """Transformer over temporal patches built only from historical cycles."""

    def __init__(
        self,
        input_dim: int,
        window_length: int = 50,
        patch_length: int = 5,
        patch_stride: int = 5,
        projection_dim: int = 64,
        layers: int = 2,
        heads: int = 4,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        positional_encoding: str = "sinusoidal",
        pooling: str = "mean",
        causal_attention: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        if input_dim <= 1:
            raise ValueError("input_dim must include at least one feature plus mask.")
        if window_length <= 0 or patch_length <= 0 or patch_stride <= 0:
            raise ValueError("window_length, patch_length, and patch_stride must be positive.")
        if patch_length > window_length:
            raise ValueError("patch_length must not exceed window_length.")
        if projection_dim % heads != 0:
            raise ValueError("projection_dim must be divisible by heads.")
        if layers <= 0 or heads <= 0 or feedforward_dim <= 0:
            raise ValueError("Transformer dimensions and layer counts must be positive.")
        self.input_dim = int(input_dim)
        self.window_length = int(window_length)
        self.patch_length = int(patch_length)
        self.patch_stride = int(patch_stride)
        self.projection_dim = int(projection_dim)
        self.layers = int(layers)
        self.heads = int(heads)
        self.feedforward_dim = int(feedforward_dim)
        self.causal_attention = bool(causal_attention)
        self.patch_starts = list(range(0, self.window_length, self.patch_stride))
        self.patch_count = len(self.patch_starts)
        self.patch_projection = nn.Linear(input_dim, projection_dim)
        self.positional = build_positional_encoding(positional_encoding, projection_dim, self.patch_count)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=projection_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        encoder_kwargs = {}
        if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder).parameters:
            encoder_kwargs["enable_nested_tensor"] = False
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers, **encoder_kwargs)
        self.pool = build_pooling(pooling, projection_dim)
        self.head = PositiveHead(projection_dim, projection_dim, dropout)
        self.model_kind = "patch_transformer"

    def extract_patches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != self.window_length:
            raise ValueError("Expected [batch, window_length, input_dim] tensor.")
        features, mask = split_features_mask(x)
        patch_vectors = []
        patch_masks = []
        validity_fractions = []
        for start in self.patch_starts:
            stop = min(start + self.patch_length, self.window_length)
            feature_slice = features[:, start:stop, :]
            mask_slice = mask[:, start:stop, :]
            valid_counts = mask_slice.sum(dim=1).clamp_min(0.0)
            patch_valid = valid_counts.squeeze(-1) > 0
            weighted_mean = (feature_slice * mask_slice).sum(dim=1) / valid_counts.clamp_min(1.0)
            fraction = valid_counts / float(stop - start)
            patch_vectors.append(torch.cat([weighted_mean, fraction], dim=1))
            patch_masks.append(patch_valid)
            validity_fractions.append(fraction.squeeze(-1))
        patches = torch.stack(patch_vectors, dim=1)
        patch_mask = torch.stack(patch_masks, dim=1)
        fractions = torch.stack(validity_fractions, dim=1)
        validate_token_mask(patch_mask)
        return patches, patch_mask, fractions

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor | None:
        if not self.causal_attention:
            return None
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def encode_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches, patch_mask, fractions = self.extract_patches(x)
        tokens = self.patch_projection(patches)
        tokens = self.positional(tokens)
        encoded = self.encoder(
            tokens,
            mask=self._causal_mask(tokens.shape[1], tokens.device),
            src_key_padding_mask=~patch_mask,
        )
        encoded = encoded * patch_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        return encoded, patch_mask, fractions

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        encoded, patch_mask, _ = self.encode_tokens(x)
        return self.head(self.pool(encoded, patch_mask))

    def patch_metadata(self) -> dict[str, int | float]:
        token_count = int(self.patch_count)
        return {
            "patch_length": int(self.patch_length),
            "patch_stride": int(self.patch_stride),
            "patch_token_count": token_count,
            "patch_coverage_cycles": int(min(self.window_length, self.patch_starts[-1] + self.patch_length)),
            "attention_complexity_scale": int(self.layers * self.heads * token_count * token_count),
            "cycle_attention_complexity_equivalent": int(self.layers * self.heads * self.window_length * self.window_length),
            "attention_scale_ratio": float((token_count * token_count) / max(self.window_length * self.window_length, 1)),
        }
