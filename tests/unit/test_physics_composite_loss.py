import pytest
import torch

from aeroguard.deep.physics.composite_loss import CompositePhysicsLoss


def _outputs() -> dict[str, torch.Tensor]:
    rul_raw = torch.tensor([[10.0], [12.0], [8.0], [7.0]], requires_grad=True)
    latent = torch.randn(4, 3, requires_grad=True)
    return {
        "rul_raw": rul_raw,
        "rul_prediction": torch.nn.functional.softplus(rul_raw),
        "health_score": torch.sigmoid(torch.tensor([[1.0], [0.5], [0.2], [0.0]], requires_grad=True)),
        "degradation_rate": torch.nn.functional.softplus(torch.tensor([[1.0], [1.0], [1.0], [1.0]], requires_grad=True)),
        "latent": latent,
    }


def _batch() -> dict[str, torch.Tensor]:
    return {
        "target_rul": torch.tensor([[10.0], [9.0], [8.0], [7.0]]),
        "health_target": torch.tensor([[1.0], [0.8], [0.4], [0.2]]),
        "pair_indices": torch.tensor([[0, 1], [1, 2]]),
        "pair_cycle_gaps": torch.tensor([[1.0], [1.0]]),
        "pair_plateau_mask": torch.tensor([[0.0], [0.0]]),
        "triplet_indices": torch.tensor([[0, 1, 2]]),
        "triplet_left_gaps": torch.tensor([[1.0]]),
        "triplet_right_gaps": torch.tensor([[1.0]]),
        "regime_pair_indices": torch.tensor([[0, 3]]),
    }


def test_data_only_candidate() -> None:
    result = CompositePhysicsLoss({"lambda_data": 1.0})(_outputs(), {"target_rul": _batch()["target_rul"]})

    assert torch.isfinite(result["total_loss"])
    assert result["weighted_terms"]["monotonic_loss"].item() == 0.0


def test_full_combined_loss_and_gradient_flow() -> None:
    outputs = _outputs()
    loss_fn = CompositePhysicsLoss(
        {
            "lambda_data": 1.0,
            "lambda_monotonic": 0.1,
            "lambda_rate": 0.1,
            "lambda_smooth": 0.1,
            "lambda_health": 0.1,
            "lambda_health_monotonic": 0.1,
            "lambda_regime": 0.1,
            "lambda_nonnegative": 0.1,
            "lambda_optimistic": 0.1,
            "include_rate_head_loss": True,
        }
    )

    result = loss_fn(outputs, _batch())
    result["total_loss"].backward()

    assert torch.isfinite(result["total_loss"])
    assert outputs["rul_raw"].grad is not None
    assert result["raw_terms"]["safety_optimistic_loss"].item() >= 0.0


def test_disabled_zero_weight_term() -> None:
    result = CompositePhysicsLoss({"lambda_data": 1.0, "lambda_monotonic": 0.0})(_outputs(), {"target_rul": _batch()["target_rul"]})

    assert result["weighted_terms"]["monotonic_loss"].item() == 0.0


def test_invalid_weight() -> None:
    with pytest.raises(ValueError, match="lambda_monotonic"):
        CompositePhysicsLoss({"lambda_data": 1.0, "lambda_monotonic": -1.0})


def test_missing_required_pair_input() -> None:
    with pytest.raises(ValueError, match="monotonic_loss"):
        CompositePhysicsLoss({"lambda_data": 1.0, "lambda_monotonic": 0.1})(_outputs(), {"target_rul": _batch()["target_rul"]})
