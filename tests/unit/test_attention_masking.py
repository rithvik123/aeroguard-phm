import pytest
import torch

from aeroguard.deep.models.attention_pooling import AttentionPooling, final_valid_token_pool, masked_mean_pool


def test_masked_mean_ignores_padded_values() -> None:
    tokens = torch.tensor([[[100.0, 100.0], [1.0, 3.0], [3.0, 5.0]]])
    mask = torch.tensor([[False, True, True]])

    pooled = masked_mean_pool(tokens, mask)

    torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0]]))


def test_final_valid_token_handles_left_padding() -> None:
    tokens = torch.tensor([[[99.0], [1.0], [2.0], [3.0]]])
    mask = torch.tensor([[False, True, True, True]])

    torch.testing.assert_close(final_valid_token_pool(tokens, mask), torch.tensor([[3.0]]))


def test_attention_pooling_assigns_zero_weight_to_padding_and_rejects_all_padding() -> None:
    pool = AttentionPooling(model_dim=2)
    tokens = torch.randn(1, 3, 2)
    mask = torch.tensor([[True, False, True]])

    weights = pool.attention_weights(tokens, mask)

    assert weights[0, 1].item() == pytest.approx(0.0)
    assert weights.sum().item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="All-padding"):
        pool(tokens, torch.zeros(1, 3, dtype=torch.bool))

