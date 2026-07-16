import torch
import pytest

from aeroguard.deep.models.positional_encoding import LearnablePositionalEncoding, SinusoidalPositionalEncoding


def test_sinusoidal_positional_encoding_shape_and_determinism() -> None:
    enc = SinusoidalPositionalEncoding(model_dim=4, max_length=8)
    x = torch.zeros(2, 3, 4)

    y1 = enc(x)
    y2 = enc(x)

    assert tuple(y1.shape) == (2, 3, 4)
    torch.testing.assert_close(y1, y2)
    torch.testing.assert_close(y1[0, 0], torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_positional_encoding_rejects_length_over_capacity() -> None:
    enc = SinusoidalPositionalEncoding(model_dim=4, max_length=2)

    with pytest.raises(ValueError, match="capacity"):
        enc(torch.zeros(1, 3, 4))


def test_learnable_positional_encoding_has_gradient_flow() -> None:
    enc = LearnablePositionalEncoding(model_dim=4, max_length=5)
    y = enc(torch.zeros(1, 3, 4)).sum()
    y.backward()

    assert enc.embedding.weight.grad is not None
    assert torch.isfinite(enc.embedding.weight.grad).all()

