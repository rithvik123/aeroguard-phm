"""Compact temporal Transformer encoder for RUL prediction."""

from __future__ import annotations

import inspect

import torch
from torch import nn

from aeroguard.deep.models.attention_pooling import build_pooling, validate_token_mask
from aeroguard.deep.models.common import PositiveHead, split_features_mask
from aeroguard.deep.models.positional_encoding import build_positional_encoding


class TemporalTransformerRegressor(nn.Module):
    """Transformer over past-only cycle tokens with padding-aware pooling."""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 64,
        layers: int = 2,
        heads: int = 4,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        positional_encoding: str = "sinusoidal",
        pooling: str = "mean",
        max_length: int = 128,
        causal_attention: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        if input_dim <= 1:
            raise ValueError("input_dim must include at least one feature plus mask.")
        if projection_dim <= 0 or layers <= 0 or heads <= 0 or feedforward_dim <= 0:
            raise ValueError("Transformer dimensions and layer counts must be positive.")
        if projection_dim % heads != 0:
            raise ValueError("projection_dim must be divisible by heads.")
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        self.input_dim = int(input_dim)
        self.projection_dim = int(projection_dim)
        self.layers = int(layers)
        self.heads = int(heads)
        self.feedforward_dim = int(feedforward_dim)
        self.max_length = int(max_length)
        self.causal_attention = bool(causal_attention)
        self.input_projection = nn.Linear(input_dim - 1, projection_dim)
        self.positional = build_positional_encoding(positional_encoding, projection_dim, max_length)
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
        self.model_kind = "temporal_transformer"

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor | None:
        if not self.causal_attention:
            return None
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def encode_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features, mask = split_features_mask(x)
        token_mask = validate_token_mask(mask)
        if x.shape[1] > self.max_length:
            raise ValueError("Sequence length exceeds configured positional capacity.")
        tokens = self.input_projection(features)
        tokens = self.positional(tokens)
        encoded = self.encoder(
            tokens,
            mask=self._causal_mask(tokens.shape[1], tokens.device),
            src_key_padding_mask=~token_mask,
        )
        encoded = encoded * token_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        return encoded, token_mask

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        encoded, token_mask = self.encode_tokens(x)
        return self.head(self.pool(encoded, token_mask))

    def attention_complexity(self, sequence_length: int | None = None) -> int:
        length = int(sequence_length or self.max_length)
        return int(self.layers * self.heads * length * length)
