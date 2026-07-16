"""Model size and inference-efficiency measurements."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aeroguard.deep.models.common import trainable_parameter_count


def serialized_state_size_bytes(model: torch.nn.Module) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        torch.save(model.state_dict(), path)
        return int(path.stat().st_size)


@torch.no_grad()
def latency_summary(model: torch.nn.Module, example: torch.Tensor, device: torch.device, repetitions: int = 100) -> dict[str, float | None]:
    model = model.to(device).eval()
    x = example.to(device)
    lengths = x[:, :, -1].sum(dim=1).long().clamp_min(1)
    for _ in range(min(10, repetitions)):
        model(x, lengths)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    for _ in range(int(repetitions)):
        start = time.perf_counter()
        model(x, lengths)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
    return {
        "median_latency_ms": float(np.median(timings) * 1000.0),
        "p95_latency_ms": float(np.quantile(timings, 0.95) * 1000.0),
        "throughput_per_second": float(len(x) / np.mean(timings)),
    }


def model_efficiency_row(
    model_id: str,
    model: torch.nn.Module,
    example_single: torch.Tensor,
    example_batch: torch.Tensor,
    device: torch.device,
    training_metadata: dict[str, Any],
    repetitions: int = 100,
) -> dict[str, Any]:
    cpu = torch.device("cpu")
    model_cpu = model.to(cpu)
    single_cpu = latency_summary(model_cpu, example_single.cpu(), cpu, repetitions)
    batch_cpu = latency_summary(model_cpu, example_batch.cpu(), cpu, max(20, repetitions // 5))
    row = {
        "model_id": model_id,
        "parameter_count": trainable_parameter_count(model),
        "serialized_size_bytes": serialized_state_size_bytes(model_cpu),
        "training_runtime_seconds": training_metadata.get("training_seconds"),
        "mean_epoch_runtime_seconds": training_metadata.get("mean_epoch_seconds"),
        "best_epoch": training_metadata.get("best_epoch"),
        "cpu_batch_one_median_latency_ms": single_cpu["median_latency_ms"],
        "cpu_batch_one_p95_latency_ms": single_cpu["p95_latency_ms"],
        "cpu_batch_32_median_latency_ms": batch_cpu["median_latency_ms"],
        "cpu_batch_32_p95_latency_ms": batch_cpu["p95_latency_ms"],
        "cpu_throughput_per_second": batch_cpu["throughput_per_second"],
        "peak_device_memory_bytes": training_metadata.get("peak_device_memory_bytes"),
    }
    if device.type == "cuda":
        model_gpu = model.to(device)
        gpu_single = latency_summary(model_gpu, example_single.to(device), device, max(20, repetitions // 5))
        gpu_batch = latency_summary(model_gpu, example_batch.to(device), device, max(20, repetitions // 5))
        row.update(
            {
                "gpu_batch_one_median_latency_ms": gpu_single["median_latency_ms"],
                "gpu_batch_one_p95_latency_ms": gpu_single["p95_latency_ms"],
                "gpu_batch_32_median_latency_ms": gpu_batch["median_latency_ms"],
                "gpu_batch_32_p95_latency_ms": gpu_batch["p95_latency_ms"],
                "gpu_throughput_per_second": gpu_batch["throughput_per_second"],
            }
        )
    else:
        row.update({"gpu_batch_one_median_latency_ms": None, "gpu_batch_one_p95_latency_ms": None, "gpu_batch_32_median_latency_ms": None, "gpu_batch_32_p95_latency_ms": None, "gpu_throughput_per_second": None})
    return row
