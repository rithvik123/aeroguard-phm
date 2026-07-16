import pytest
import torch

from aeroguard.deep.physics.health_targets import normalized_capped_rul_targets, normalized_life_fraction_targets, validate_health_range


def test_normalized_capped_rul_targets() -> None:
    target = normalized_capped_rul_targets([0.0, 50.0, 125.0, 150.0], 125.0)

    assert torch.all((target >= 0.0) & (target <= 1.0))
    assert target[-1].item() == 1.0


def test_normalized_life_fraction_targets() -> None:
    target = normalized_life_fraction_targets([1, 5, 9], [9, 9, 9])

    assert target[0].item() == 1.0
    assert target[-1].item() == 0.0


def test_life_fraction_requires_full_run_to_failure() -> None:
    with pytest.raises(ValueError, match="full run-to-failure"):
        normalized_life_fraction_targets([1], [10], full_run_to_failure=False)


def test_health_range_validation() -> None:
    validate_health_range(torch.tensor([[0.0], [1.0]]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_health_range(torch.tensor([[1.2]]))


def test_health_targets_reject_future_benchmark_style_cycle() -> None:
    with pytest.raises(ValueError, match="no greater"):
        normalized_life_fraction_targets([11], [10])
