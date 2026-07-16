"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(payload["state_dict"])
    return dict(payload.get("metadata", {}))

