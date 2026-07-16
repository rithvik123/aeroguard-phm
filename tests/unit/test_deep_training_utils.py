import numpy as np
import pytest
import torch

from aeroguard.deep.early_stopping import EarlyStopping
from aeroguard.deep.models.sequence_mlp import SequenceMLPRegressor
from aeroguard.deep.sequence_dataset import SequenceWindowDataset
from aeroguard.deep.training import make_loss, train_fixed_epochs, train_model


def _dataset() -> SequenceWindowDataset:
    sequences = np.random.default_rng(3).normal(size=(6, 4, 3)).astype(np.float32)
    sequences[:, :, -1] = 1.0
    targets = np.linspace(1.0, 6.0, 6, dtype=np.float32)
    return SequenceWindowDataset(sequences, targets)


def _config() -> dict:
    return {
        "optimizer": "adam",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "loss": "mse",
        "batch_size": 3,
        "num_workers": 0,
        "pin_memory": False,
        "gradient_clip_norm": 5.0,
    }


def test_make_loss_and_early_stopping() -> None:
    assert make_loss("mse").__class__.__name__ == "MSELoss"
    assert make_loss("smooth_l1").__class__.__name__ == "SmoothL1Loss"
    with pytest.raises(ValueError, match="Unsupported loss"):
        make_loss("mae")

    stopper = EarlyStopping(patience=1)
    assert stopper.update(3.0, 1) is True
    assert stopper.update(3.5, 2) is False
    assert stopper.should_stop is False
    assert stopper.update(3.6, 3) is False
    assert stopper.should_stop is True


def test_train_model_and_fixed_epoch_training_return_metadata() -> None:
    train = _dataset()
    validation = _dataset()
    device = torch.device("cpu")

    model, metadata = train_model(
        SequenceMLPRegressor(input_dim=3, hidden_dim=6, dropout=0.0),
        train,
        validation,
        _config(),
        device,
        max_epochs=1,
        patience=0,
        mixed_precision=False,
    )
    assert len(metadata["history"]) == 1
    assert metadata["best_epoch"] == 1

    _, fixed_metadata = train_fixed_epochs(model, train, _config(), device, epochs=1, mixed_precision=False)
    assert fixed_metadata["early_stopping_reason"] == "locked_epoch_count"
    assert len(fixed_metadata["history"]) == 1

