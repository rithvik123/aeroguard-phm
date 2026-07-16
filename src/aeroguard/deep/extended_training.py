"""Extended bounded training utilities for Phase 5B temporal models."""

from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from aeroguard.deep.inference import predict_batches
from aeroguard.deep.sequence_dataset import SequenceWindowDataset
from aeroguard.deep.training import make_loss
from aeroguard.evaluation.deep_rul_metrics import deep_point_metrics
from aeroguard.evaluation.metrics import nasa_asymmetric_score


def engine_balanced_validation_metrics(
    true: np.ndarray,
    predicted: np.ndarray,
    metadata: pd.DataFrame | None = None,
    severe_optimistic_threshold: float = 30.0,
) -> dict[str, float]:
    if metadata is None or "global_engine_id" not in metadata.columns:
        return deep_point_metrics(true, predicted, severe_optimistic_threshold)
    frame = pd.DataFrame(
        {
            "engine": metadata["global_engine_id"].to_numpy(),
            "true": np.asarray(true, dtype=float),
            "pred": np.asarray(predicted, dtype=float),
        }
    )
    rows = []
    for _, group in frame.groupby("engine"):
        residual = group["pred"].to_numpy(dtype=float) - group["true"].to_numpy(dtype=float)
        rows.append(
            {
                "mae": float(np.mean(np.abs(residual))),
                "mse": float(np.mean(np.square(residual))),
                "nasa_score": nasa_asymmetric_score(group["true"], group["pred"]),
                "mean_signed_error": float(np.mean(residual)),
                "optimistic_prediction_rate": float((residual > 0).mean()),
                "conservative_prediction_rate": float((residual < 0).mean()),
                "severe_optimistic_error_rate": float((residual > severe_optimistic_threshold).mean()),
            }
        )
    metrics = pd.DataFrame(rows)
    return {
        "mae": float(metrics["mae"].mean()),
        "rmse": float(np.sqrt(metrics["mse"].mean())),
        "nasa_score": float(metrics["nasa_score"].mean()),
        "mean_signed_error": float(metrics["mean_signed_error"].mean()),
        "optimistic_prediction_rate": float(metrics["optimistic_prediction_rate"].mean()),
        "conservative_prediction_rate": float(metrics["conservative_prediction_rate"].mean()),
        "severe_optimistic_error_rate": float(metrics["severe_optimistic_error_rate"].mean()),
    }


def _optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "adamw")).lower()
    lr = float(config["learning_rate"])
    weight_decay = float(config.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def _scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any], max_epochs: int) -> Any:
    name = str(config.get("scheduler", "none")).lower()
    if name == "none":
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.get("scheduler_factor", 0.5)),
            patience=int(config.get("scheduler_patience", 3)),
            min_lr=float(config.get("min_learning_rate", 1.0e-6)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(max_epochs)),
            eta_min=float(config.get("min_learning_rate", 1.0e-6)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def _validate_training_config(config: dict[str, Any]) -> None:
    if int(config["max_epochs"]) <= 0:
        raise ValueError("max_epochs must be positive.")
    if int(config["minimum_epochs"]) <= 0 or int(config["minimum_epochs"]) > int(config["max_epochs"]):
        raise ValueError("minimum_epochs must be in [1, max_epochs].")
    if int(config["early_stopping_patience"]) < 0:
        raise ValueError("early_stopping_patience must be non-negative.")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive.")


def train_with_early_stopping(
    model: nn.Module,
    train_dataset: SequenceWindowDataset,
    validation_dataset: SequenceWindowDataset,
    config: dict[str, Any],
    device: torch.device,
    validation_metadata: pd.DataFrame | None = None,
    mixed_precision: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train with minimum epochs, patience, scheduler updates, and best restoration."""

    _validate_training_config(config)
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Training and validation datasets must not be empty.")
    max_epochs = int(config["max_epochs"])
    minimum_epochs = int(config["minimum_epochs"])
    patience = int(config["early_stopping_patience"])
    model = model.to(device)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config, max_epochs)
    loss_fn = make_loss(str(config.get("loss", "smooth_l1")))
    loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(mixed_precision and device.type == "cuda"))
    best_state = deepcopy(model.state_dict())
    best_rmse = math.inf
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float | int | bool]] = []
    start = time.perf_counter()
    peak_memory = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        grad_norms = []
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
            grad_norms.append(float(grad_norm.detach().cpu()))
        pred = predict_batches(model, validation_dataset, device, int(config["batch_size"]))
        true = validation_dataset.targets.detach().cpu().numpy().ravel()
        val_loss = float(loss_fn(torch.as_tensor(pred).view(-1, 1), torch.as_tensor(true).view(-1, 1)).item())
        metrics = engine_balanced_validation_metrics(true, pred, validation_metadata, float(config.get("severe_optimistic_threshold", 30.0)))
        current_rmse = float(metrics["rmse"])
        improved = current_rmse < best_rmse - float(config.get("min_delta", 0.0))
        if improved:
            best_rmse = current_rmse
            best_epoch = epoch
            bad_epochs = 0
            best_state = deepcopy(model.state_dict())
        elif epoch >= minimum_epochs:
            bad_epochs += 1
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_rmse)
            else:
                scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": val_loss,
            "validation_mae": float(metrics["mae"]),
            "validation_rmse": current_rmse,
            "validation_nasa_score": float(metrics["nasa_score"]),
            "validation_mean_signed_error": float(metrics["mean_signed_error"]),
            "validation_optimistic_rate": float(metrics["optimistic_prediction_rate"]),
            "validation_severe_optimistic_rate": float(metrics["severe_optimistic_error_rate"]),
            "gradient_norm": float(np.mean(grad_norms)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_start,
            "improved": bool(improved),
        }
        history.append(row)
        if epoch >= minimum_epochs and bad_epochs > patience:
            break
    model.load_state_dict(best_state)
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    best = min(history, key=lambda item: float(item["validation_rmse"]))
    metadata = {
        "history": history,
        "best_epoch": int(best_epoch or best["epoch"]),
        "stopping_epoch": int(history[-1]["epoch"]),
        "best_validation_rmse": float(best["validation_rmse"]),
        "best_validation_mae": float(best["validation_mae"]),
        "best_validation_nasa_score": float(best["validation_nasa_score"]),
        "best_validation_optimistic_rate": float(best["validation_optimistic_rate"]),
        "best_validation_severe_optimistic_rate": float(best["validation_severe_optimistic_rate"]),
        "early_stopping_triggered": bool(int(history[-1]["epoch"]) < max_epochs),
        "early_stopping_reason": "patience_exhausted" if int(history[-1]["epoch"]) < max_epochs else "max_epochs_reached",
        "training_seconds": time.perf_counter() - start,
        "mean_epoch_seconds": float(np.mean([float(row["epoch_seconds"]) for row in history])),
        "peak_device_memory_bytes": peak_memory,
        "mixed_precision_used": bool(mixed_precision and device.type == "cuda"),
        "scheduler": str(config.get("scheduler", "none")),
    }
    return model, metadata


def train_for_fixed_epochs(
    model: nn.Module,
    train_dataset: SequenceWindowDataset,
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    mixed_precision: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train a locked model for exactly the selected epoch count."""

    locked_config = dict(config)
    locked_config["max_epochs"] = int(epochs)
    locked_config["minimum_epochs"] = int(epochs)
    locked_config["early_stopping_patience"] = int(epochs)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset must not be empty.")
    model = model.to(device)
    optimizer = _optimizer(model, locked_config)
    scheduler = _scheduler(optimizer, locked_config, int(epochs))
    loss_fn = make_loss(str(locked_config.get("loss", "smooth_l1")))
    loader = DataLoader(train_dataset, batch_size=int(locked_config["batch_size"]), shuffle=True, num_workers=int(locked_config.get("num_workers", 0)), pin_memory=bool(locked_config.get("pin_memory", False)))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(mixed_precision and device.type == "cuda"))
    history = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(epochs) + 1):
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        grad_norms = []
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
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(locked_config["gradient_clip_norm"]))
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite gradients encountered.")
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(float(grad_norm.detach().cpu()))
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "gradient_norm": float(np.mean(grad_norms)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    return model, {
        "history": history,
        "best_epoch": int(epochs),
        "stopping_epoch": int(epochs),
        "early_stopping_reason": "locked_epoch_count",
        "early_stopping_triggered": False,
        "training_seconds": time.perf_counter() - start,
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
        "peak_device_memory_bytes": peak_memory,
        "mixed_precision_used": bool(mixed_precision and device.type == "cuda"),
        "scheduler": str(locked_config.get("scheduler", "none")),
    }

