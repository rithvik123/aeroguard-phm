import pytest
import torch

from aeroguard.deep.physics.temporal_constraints import cycle_rate_consistency_loss, monotonicity_loss, smoothness_loss


def test_monotonicity_no_violation_and_positive_violation() -> None:
    earlier = torch.tensor([[10.0], [8.0]])
    later = torch.tensor([[9.0], [9.5]])

    assert monotonicity_loss(earlier[:1], later[:1]).item() == 0.0
    assert monotonicity_loss(earlier, later, reduction="none")[1].item() > 0.0


def test_monotonicity_tolerance_and_gap_normalization() -> None:
    earlier = torch.tensor([[10.0]])
    later = torch.tensor([[13.0]])

    assert monotonicity_loss(earlier, later, tolerance=3.0).item() == 0.0
    assert monotonicity_loss(earlier, later, cycle_gap=torch.tensor([2.0]), normalize_by_gap=True).item() == pytest.approx(1.5)


def test_monotonicity_reductions() -> None:
    earlier = torch.tensor([1.0, 1.0])
    later = torch.tensor([2.0, 3.0])

    assert monotonicity_loss(earlier, later, reduction="sum").item() == pytest.approx(3.0)
    assert tuple(monotonicity_loss(earlier, later, reduction="none").shape) == (2, 1)


def test_cycle_rate_perfect_incorrect_and_multiple_gaps() -> None:
    earlier = torch.tensor([[10.0], [20.0]])
    later = torch.tensor([[9.0], [17.0]])
    gaps = torch.tensor([[1.0], [3.0]])

    assert cycle_rate_consistency_loss(earlier, later, gaps, kind="absolute").item() == 0.0
    assert cycle_rate_consistency_loss(earlier, later + 2.0, gaps, kind="absolute").item() > 0.0


def test_cycle_rate_tolerance_and_invalid_gap() -> None:
    earlier = torch.tensor([[10.0]])
    later = torch.tensor([[8.5]])

    assert cycle_rate_consistency_loss(earlier, later, torch.tensor([1.0]), tolerance=0.5, kind="absolute").item() == 0.0
    with pytest.raises(ValueError, match="positive"):
        cycle_rate_consistency_loss(earlier, later, torch.tensor([0.0]))


def test_smoothness_linear_curved_and_tolerance() -> None:
    earlier = torch.tensor([[10.0]])
    middle = torch.tensor([[8.0]])
    later = torch.tensor([[6.0]])
    curved = torch.tensor([[9.0]])

    assert smoothness_loss(earlier, middle, later, kind="absolute").item() == 0.0
    assert smoothness_loss(earlier, curved, later, kind="absolute").item() > 0.0
    assert smoothness_loss(earlier, curved, later, kind="absolute", tolerance=2.0).item() == 0.0


def test_smoothness_unequal_gap_rejection() -> None:
    with pytest.raises(ValueError, match="Unequal-gap"):
        smoothness_loss(
            torch.tensor([[10.0]]),
            torch.tensor([[8.0]]),
            torch.tensor([[5.0]]),
            left_gap=torch.tensor([1.0]),
            right_gap=torch.tensor([2.0]),
        )
