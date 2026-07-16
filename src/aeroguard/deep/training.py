"""Bounded PyTorch training utilities."""

from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from aeroguard.deep.early_stopping import EarlyStopping
from aeroguard.deep.inference import predict_batches
from aeroguard.deep.sequence_dataset import SequenceWindowDataset
from aeroguard.evaluation.metrics import nasa_asymmetric_score, regression_metrics


def make_loss(name: str) -> nn.Module:
    if name == "smooth_l1":
        return nn.SmoothL1Loss()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss: {name}")


def validation_metrics(model: nn.Module, dataset: SequenceWindowDataset, device: torch.device, batch_size: int) -> dict[str, float]:
    pred = predict_batches(model, dataset, device, batch_size=batch_size)
    true = dataset.targets.detach().cpu().numpy().ravel()
    metrics = regression_metrics(true, pred)
    residual = pred - true
    metrics.update(
        {
            "nasa_score": nasa_asymmetric_score(true, pred),
            "mean_signed_error": float(residual.mean()),
            "optimistic_prediction_rate": float((residual > 0).mean()),
            "conservative_prediction_rate": float((residual < 0).mean()),
        }
    )
    return metrics


def train_model(
    model: nn.Module,
    train_dataset: SequenceWindowDataset,
    validation_dataset: SequenceWindowDataset,
    config: dict[str, Any],
    device: torch.device,
    max_epochs: int,
    patience: int,
    mixed_precision: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Training and validation datasets must not be empty.")
    model = model.to(device)
    optimizer_name = str(config["optimizer"]).lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    loss_fn = make_loss(str(config["loss"]))
    loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config.get("pin_memory", False)),
    )
    stopper = EarlyStopping(int(patience))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(mixed_precision and device.type == "cuda"))
    best_state = deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    start = time.perf_counter()
    peak_memory = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(max_epochs) + 1):
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        for x, y, lengths in loader:
            if len(x) == 0:
                raise ValueError("Empty batch encountered.")
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(mixed_precision and device.type == "cuda")):
                pred = model(x, lengths)
                loss = loss_fn(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite gradients encountered.")
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        val = validation_metrics(model, validation_dataset, device, int(config["batch_size"]))
        improved = stopper.update(float(val["rmse"]), epoch)
        if improved:
            best_state = deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_mae": float(val["mae"]),
                "validation_rmse": float(val["rmse"]),
                "validation_nasa_score": float(val["nasa_score"]),
                "validation_optimistic_rate": float(val["optimistic_prediction_rate"]),
                "epoch_seconds": time.perf_counter() - epoch_start,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if stopper.should_stop:
            break
    model.load_state_dict(best_state)
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    best = min(history, key=lambda item: item["validation_rmse"])
    metadata = {
        "history": history,
        "best_epoch": int(best["epoch"]),
        "best_validation_rmse": float(best["validation_rmse"]),
        "best_validation_mae": float(best["validation_mae"]),
        "best_validation_nasa_score": float(best["validation_nasa_score"]),
        "early_stopping_reason": "patience_exhausted" if stopper.should_stop else "max_epochs_reached",
        "training_seconds": time.perf_counter() - start,
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])) if history else math.nan,
        "peak_device_memory_bytes": peak_memory,
        "mixed_precision_used": bool(mixed_precision and device.type == "cuda"),
    }
    return model, metadata


def train_fixed_epochs(
    model: nn.Module,
    train_dataset: SequenceWindowDataset,
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    mixed_precision: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    if len(train_dataset) == 0:
        raise ValueError("Training dataset must not be empty.")
    model = model.to(device)
    optimizer_name = str(config["optimizer"]).lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    loss_fn = make_loss(str(config["loss"]))
    loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config["num_workers"]), pin_memory=bool(config.get("pin_memory", False)))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(mixed_precision and device.type == "cuda"))
    history = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(epochs) + 1):
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        for x, y, lengths in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(mixed_precision and device.type == "cuda")):
                pred = model(x, lengths)
                loss = loss_fn(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite gradients encountered.")
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "epoch_seconds": time.perf_counter() - epoch_start, "learning_rate": float(optimizer.param_groups[0]["lr"])})
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    metadata = {
        "history": history,
        "best_epoch": int(epochs),
        "best_validation_rmse": None,
        "best_validation_mae": None,
        "best_validation_nasa_score": None,
        "early_stopping_reason": "locked_epoch_count",
        "training_seconds": time.perf_counter() - start,
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])) if history else math.nan,
        "peak_device_memory_bytes": peak_memory,
        "mixed_precision_used": bool(mixed_precision and device.type == "cuda"),
    }
    return model, metadata
