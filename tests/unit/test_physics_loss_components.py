import torch

from aeroguard.deep.physics.loss_components import (
    degradation_rate_head_loss,
    health_proxy_loss,
    nonnegative_rul_penalty,
    optimistic_prediction_loss,
    primary_rul_loss,
)


def test_primary_data_loss_variants() -> None:
    pred = torch.tensor([[1.0], [3.0]])
    true = torch.tensor([[1.0], [1.0]])

    assert primary_rul_loss(pred, true, kind="mae").item() == 1.0
    assert primary_rul_loss(pred, true, kind="mse").item() == 2.0
    assert primary_rul_loss(pred, true, kind="smooth_l1").item() > 0.0


def test_nonnegative_penalty() -> None:
    assert nonnegative_rul_penalty(torch.tensor([[-2.0], [1.0]])).item() == 1.0


def test_health_proxy_loss() -> None:
    assert health_proxy_loss(torch.tensor([[0.5]]), torch.tensor([[0.5]])).item() == 0.0


def test_degradation_rate_head_loss_masks_capped_plateau() -> None:
    loss = degradation_rate_head_loss(
        torch.tensor([[2.0], [1.0]]),
        torch.tensor([[10.0], [8.0]]),
        torch.tensor([[10.0], [6.0]]),
        torch.tensor([[1.0], [2.0]]),
        plateau_mask=torch.tensor([[1.0], [0.0]]),
    )

    assert loss.item() == 0.0


def test_optimistic_prediction_loss_is_asymmetric() -> None:
    pred = torch.tensor([[12.0], [5.0]])
    true = torch.tensor([[10.0], [8.0]])

    assert optimistic_prediction_loss(pred, true, kind="mae").item() == 1.0
    assert optimistic_prediction_loss(pred, true, kind="mse", severe_threshold=1.0, severe_multiplier=2.0).item() == 4.0
