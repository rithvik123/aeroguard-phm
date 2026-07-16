"""Positional encodings for temporal RUL Transformer models."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Deterministic sinusoidal positions in historical order."""

    def __init__(self, model_dim: int, max_length: int = 512) -> None:
        super().__init__()
        if model_dim <= 0 or max_length <= 0:
            raise ValueError("model_dim and max_length must be positive.")
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / model_dim))
        encoding = torch.zeros(max_length, model_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
        self.model_dim = int(model_dim)
        self.max_length = int(max_length)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected [batch, sequence, model_dim] tensor.")
        if x.shape[-1] != self.model_dim:
            raise ValueError("Input model dimension does not match positional encoding.")
        if x.shape[1] > self.max_length:
            raise ValueError("Sequence length exceeds positional encoding capacity.")
        return x + self.encoding[:, : x.shape[1], :].to(dtype=x.dtype, device=x.device)


class LearnablePositionalEncoding(nn.Module):
    """Learned absolute positions with bounded capacity."""

    def __init__(self, model_dim: int, max_length: int = 512) -> None:
        super().__init__()
        if model_dim <= 0 or max_length <= 0:
            raise ValueError("model_dim and max_length must be positive.")
        self.model_dim = int(model_dim)
        self.max_length = int(max_length)
        self.embedding = nn.Embedding(max_length, model_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected [batch, sequence, model_dim] tensor.")
        if x.shape[-1] != self.model_dim:
            raise ValueError("Input model dimension does not match positional encoding.")
        if x.shape[1] > self.max_length:
            raise ValueError("Sequence length exceeds positional encoding capacity.")
        positions = torch.arange(x.shape[1], device=x.device)
        return x + self.embedding(positions).unsqueeze(0).to(dtype=x.dtype)


def build_positional_encoding(kind: str, model_dim: int, max_length: int) -> nn.Module:
    """Build a configured positional encoding module."""

    normalized = str(kind).lower()
    if normalized == "sinusoidal":
        return SinusoidalPositionalEncoding(model_dim, max_length)
    if normalized == "learnable":
        return LearnablePositionalEncoding(model_dim, max_length)
    raise ValueError(f"Unsupported positional encoding: {kind}")

