import pytest
import torch

from aeroguard.deep.models.common import trainable_parameter_count, validate_parameter_budget
from aeroguard.deep.models.temporal_transformer import TemporalTransformerRegressor


def _input(batch: int = 2) -> torch.Tensor:
    x = torch.randn(batch, 6, 5)
    x[:, :2, -1] = 0.0
    x[:, 2:, -1] = 1.0
    return x


def test_temporal_transformer_shape_batch_one_and_finite_output() -> None:
    model = TemporalTransformerRegressor(input_dim=5, projection_dim=16, layers=1, heads=4, feedforward_dim=32, dropout=0.0, max_length=6)

    y = model(_input(1))

    assert tuple(y.shape) == (1, 1)
    assert torch.isfinite(y).all()
    validate_parameter_budget(model, trainable_parameter_count(model))


def test_temporal_transformer_padding_value_invariance() -> None:
    torch.manual_seed(3)
    model = TemporalTransformerRegressor(input_dim=5, projection_dim=16, layers=1, heads=4, feedforward_dim=32, dropout=0.0, max_length=6).eval()
    x = _input(2)
    altered = x.clone()
    altered[:, :2, :-1] += 1000.0

    torch.testing.assert_close(model(x), model(altered), rtol=1e-5, atol=1e-5)


def test_temporal_transformer_validates_head_dimension() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TemporalTransformerRegressor(input_dim=5, projection_dim=10, heads=4)

