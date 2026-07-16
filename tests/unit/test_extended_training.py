import numpy as np
import pandas as pd
import pytest
import torch

from aeroguard.deep.extended_training import train_with_early_stopping
from aeroguard.deep.models.sequence_mlp import SequenceMLPRegressor
from aeroguard.deep.sequence_dataset import SequenceWindowDataset


def _dataset() -> SequenceWindowDataset:
    rng = np.random.default_rng(9)
    sequences = rng.normal(size=(8, 4, 4)).astype(np.float32)
    sequences[:, :, -1] = 1.0
    return SequenceWindowDataset(sequences, np.linspace(1, 8, 8, dtype=np.float32))


def _config() -> dict:
    return {
        "optimizer": "adamw",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "loss": "mse",
        "batch_size": 4,
        "num_workers": 0,
        "pin_memory": False,
        "gradient_clip_norm": 1.0,
        "max_epochs": 3,
        "minimum_epochs": 2,
        "early_stopping_patience": 0,
        "scheduler": "plateau",
        "scheduler_patience": 0,
        "scheduler_factor": 0.5,
        "min_learning_rate": 1e-6,
        "severe_optimistic_threshold": 5,
    }


def test_extended_training_tracks_history_and_restores_best_checkpoint() -> None:
    dataset = _dataset()
    metadata = pd.DataFrame({"global_engine_id": [f"e{i // 2}" for i in range(len(dataset))]})

    model, meta = train_with_early_stopping(
        SequenceMLPRegressor(input_dim=4, hidden_dim=6, dropout=0.0),
        dataset,
        dataset,
        _config(),
        torch.device("cpu"),
        metadata,
    )

    assert 2 <= meta["stopping_epoch"] <= 3
    assert 1 <= meta["best_epoch"] <= meta["stopping_epoch"]
    assert {"validation_loss", "learning_rate", "gradient_norm"}.issubset(meta["history"][0])
    assert torch.isfinite(model(dataset[0][0].unsqueeze(0), dataset[0][2].view(1))).all()


def test_extended_training_rejects_nan_inputs_and_bad_config() -> None:
    bad = np.zeros((1, 4, 4), dtype=np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SequenceWindowDataset(bad, np.array([1.0], dtype=np.float32))

    config = _config()
    config["minimum_epochs"] = 5
    with pytest.raises(ValueError, match="minimum_epochs"):
        train_with_early_stopping(
            SequenceMLPRegressor(input_dim=4, hidden_dim=6, dropout=0.0),
            _dataset(),
            _dataset(),
            config,
            torch.device("cpu"),
        )

