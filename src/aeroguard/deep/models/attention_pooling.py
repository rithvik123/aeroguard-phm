"""Mask-aware pooling blocks for temporal sequence models."""

from __future__ import annotations

import torch
from torch import nn


def validate_token_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.ndim != 2:
        raise ValueError("Token mask must have shape [batch, tokens].")
    mask = mask.bool()
    if (~mask.any(dim=1)).any():
        raise ValueError("All-padding sequences cannot be pooled.")
    return mask


def masked_mean_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, tokens, features].")
    mask = validate_token_mask(mask)
    if tokens.shape[:2] != mask.shape:
        raise ValueError("Token and mask shapes are incompatible.")
    weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def final_valid_token_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, tokens, features].")
    mask = validate_token_mask(mask)
    if tokens.shape[:2] != mask.shape:
        raise ValueError("Token and mask shapes are incompatible.")
    positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand_as(mask)
    last_indices = positions.masked_fill(~mask, 0).max(dim=1).values
    return tokens[torch.arange(tokens.shape[0], device=tokens.device), last_indices]


class AttentionPooling(nn.Module):
    """Learn a scalar attention score over valid tokens only."""

    def __init__(self, model_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be positive.")
        hidden = int(hidden_dim or model_dim)
        self.scorer = nn.Sequential(nn.Linear(model_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def attention_weights(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, tokens, features].")
        mask = validate_token_mask(mask)
        if tokens.shape[:2] != mask.shape:
            raise ValueError("Token and mask shapes are incompatible.")
        scores = self.scorer(tokens).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return weights * mask.to(dtype=weights.dtype)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = self.attention_weights(tokens, mask)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        return (tokens * (weights / denom).unsqueeze(-1)).sum(dim=1)


def build_pooling(kind: str, model_dim: int) -> nn.Module:
    normalized = str(kind).lower()
    if normalized == "mean":
        return _MeanPooling()
    if normalized == "final":
        return _FinalTokenPooling()
    if normalized == "attention":
        return AttentionPooling(model_dim)
    raise ValueError(f"Unsupported pooling method: {kind}")


class _MeanPooling(nn.Module):
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mean_pool(tokens, mask)


class _FinalTokenPooling(nn.Module):
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return final_valid_token_pool(tokens, mask)
