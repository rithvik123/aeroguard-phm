import pytest
import torch

from aeroguard.deep.models.patch_transformer import PatchTemporalTransformerRegressor


def _input() -> torch.Tensor:
    x = torch.randn(2, 8, 5)
    x[:, :3, -1] = 0.0
    x[:, 3:, -1] = 1.0
    return x


def test_patch_transformer_patch_extraction_and_mask() -> None:
    model = PatchTemporalTransformerRegressor(input_dim=5, window_length=8, patch_length=3, patch_stride=2, projection_dim=16, layers=1, heads=4, feedforward_dim=32, dropout=0.0)

    patches, mask, fractions = model.extract_patches(_input())

    assert tuple(patches.shape) == (2, 4, 5)
    assert tuple(mask.shape) == (2, 4)
    assert mask[0].tolist() == [False, True, True, True]
    assert torch.isfinite(fractions).all()


def test_patch_transformer_output_and_padded_invariance() -> None:
    torch.manual_seed(4)
    model = PatchTemporalTransformerRegressor(input_dim=5, window_length=8, patch_length=4, patch_stride=4, projection_dim=16, layers=1, heads=4, feedforward_dim=32, dropout=0.0).eval()
    x = _input()
    altered = x.clone()
    altered[:, :3, :-1] -= 999.0

    torch.testing.assert_close(model(x), model(altered), rtol=1e-5, atol=1e-5)


def test_patch_transformer_rejects_invalid_patch_settings() -> None:
    with pytest.raises(ValueError, match="patch_length"):
        PatchTemporalTransformerRegressor(input_dim=5, window_length=5, patch_length=6, patch_stride=1)

