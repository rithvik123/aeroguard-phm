from __future__ import annotations

import torch

from aeroguard.kan.kan_linear import KANLinear
from aeroguard.kan.pruning import prune_layer_by_quantile
from aeroguard.kan.sparse_kan import BoundedResidualKAN, DirectKANRUL
from aeroguard.kan.spline_basis import bspline_basis


def test_bspline_basis_shape_finite_and_partition() -> None:
    x = torch.linspace(-5, 5, 13)
    basis = bspline_basis(x, grid_size=5, degree=3, value_min=-5, value_max=5)
    assert basis.shape == (13, 8)
    assert torch.isfinite(basis).all()
    assert torch.allclose(basis.sum(dim=1), torch.ones(13), atol=1e-5)


def test_kan_forward_gradient_cpu_and_deterministic_initialization() -> None:
    first = KANLinear(4, 2, grid_size=5, spline_degree=3, seed=123)
    second = KANLinear(4, 2, grid_size=5, spline_degree=3, seed=123)
    for p1, p2 in zip(first.parameters(), second.parameters()):
        assert torch.allclose(p1, p2)
    x = torch.randn(6, 4, requires_grad=True)
    y = first(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_edge_contributions_sum_to_output_and_pruning_zeroes_edge() -> None:
    layer = KANLinear(3, 1, seed=7)
    x = torch.randn(5, 3)
    contrib = layer.edge_contributions(x)
    assert torch.allclose(contrib.sum(dim=2) + layer.bias, layer(x), atol=1e-6)
    mask = torch.ones_like(layer.edge_mask)
    mask[0, 1] = 0.0
    layer.apply_pruning_mask(mask)
    pruned = layer.edge_contributions(x)
    assert torch.allclose(pruned[:, 0, 1], torch.zeros(5), atol=1e-7)


def test_save_load_prediction_identity_and_parameter_count(tmp_path) -> None:
    model = BoundedResidualKAN(3, correction_bound=10.0, grid_size=5, spline_degree=3, seed=11)
    x = torch.randn(4, 3)
    base = torch.full((4,), 20.0)
    pred = model(x, base)
    path = tmp_path / "kan.pt"
    torch.save(model.state_dict(), path)
    loaded = BoundedResidualKAN(3, correction_bound=10.0, grid_size=5, spline_degree=3, seed=12)
    loaded.load_state_dict(torch.load(path, map_location="cpu"))
    assert torch.allclose(pred, loaded(x, base), atol=1e-6)
    assert model.parameter_count() > 0


def test_bounded_residual_nonnegative_and_direct_head() -> None:
    residual = BoundedResidualKAN(2, correction_bound=5.0, grid_size=5, spline_degree=3, seed=3)
    x = torch.tensor([[100.0, -100.0]])
    base = torch.tensor([1.0])
    corrected = residual(x, base)
    correction = residual.correction(x)
    assert corrected.item() >= 0.0
    assert abs(correction.item()) <= 5.0001
    direct = DirectKANRUL(2, rul_cap=125.0, grid_size=5, spline_degree=3, seed=4)
    value = direct(x)
    assert 0.0 <= value.item() <= 125.0


def test_pruning_report_and_cuda_when_available() -> None:
    layer = KANLinear(4, 1, seed=9)
    report = prune_layer_by_quantile(layer, 0.5)
    assert report.edges_before == 4
    assert 1 <= report.edges_after <= 4
    if torch.cuda.is_available():
        cuda_layer = KANLinear(4, 1, seed=9).cuda()
        x = torch.randn(3, 4, device="cuda")
        assert torch.isfinite(cuda_layer(x)).all()
