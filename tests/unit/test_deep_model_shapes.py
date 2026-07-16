import torch

from aeroguard.deep.models import MODEL_CLASSES
from aeroguard.deep.models.common import trainable_parameter_count, validate_parameter_budget


def test_all_deep_models_return_positive_batch_outputs() -> None:
    configs = {
        "sequence_mlp": {"hidden_dim": 8, "dropout": 0.0},
        "cnn1d": {"hidden_dim": 8, "dropout": 0.0, "kernel_size": 3},
        "lstm": {"hidden_dim": 8, "dropout": 0.0, "layers": 1},
        "gru": {"hidden_dim": 8, "dropout": 0.0, "layers": 1},
        "tcn": {"hidden_dim": 8, "dropout": 0.0, "kernel_size": 3, "dilations": [1, 2]},
        "cnn_lstm": {"hidden_dim": 8, "dropout": 0.0, "kernel_size": 3, "layers": 1},
    }
    x = torch.randn(3, 7, 6)
    x[:, :, -1] = 1.0
    lengths = torch.full((3,), 7, dtype=torch.long)

    for architecture, kwargs in configs.items():
        model = MODEL_CLASSES[architecture](input_dim=6, **kwargs).eval()
        y = model(x, lengths)

        assert tuple(y.shape) == (3, 1)
        assert torch.isfinite(y).all()
        assert (y >= 0).all()
        validate_parameter_budget(model, trainable_parameter_count(model))

