"""Differentiable fixed-grid B-spline basis utilities."""

from __future__ import annotations

import torch


def open_uniform_knots(n_basis: int, degree: int, value_min: float, value_max: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if n_basis <= degree:
        raise ValueError("n_basis must be greater than spline degree")
    interior_count = n_basis - degree - 1
    if interior_count > 0:
        interior = torch.linspace(value_min, value_max, interior_count + 2, device=device, dtype=dtype)[1:-1]
        return torch.cat(
            [
                torch.full((degree + 1,), value_min, device=device, dtype=dtype),
                interior,
                torch.full((degree + 1,), value_max, device=device, dtype=dtype),
            ]
        )
    return torch.cat(
        [
            torch.full((degree + 1,), value_min, device=device, dtype=dtype),
            torch.full((degree + 1,), value_max, device=device, dtype=dtype),
        ]
    )


def bspline_basis(x: torch.Tensor, *, grid_size: int = 5, degree: int = 3, value_min: float = -5.0, value_max: float = 5.0) -> torch.Tensor:
    """Return open-uniform B-spline basis values for each scalar in ``x``.

    The number of basis functions is ``grid_size + degree``. Inputs are clamped
    into the configured interval so the basis remains finite for deployment
    outliers. The returned tensor has shape ``x.shape + (n_basis,)``.
    """

    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if value_max <= value_min:
        raise ValueError("value_max must exceed value_min")

    x_clamped = torch.clamp(x, min=value_min, max=value_max)
    n_basis = int(grid_size) + int(degree)
    knots = open_uniform_knots(n_basis, degree, value_min, value_max, device=x.device, dtype=x.dtype)
    flat = x_clamped.reshape(-1, 1)

    left = knots[:-1].reshape(1, -1)
    right = knots[1:].reshape(1, -1)
    basis = ((flat >= left) & (flat < right)).to(dtype=x.dtype)
    # Include the right boundary in the final basis cell.
    basis = torch.where((flat == value_max) & (right == value_max), torch.ones_like(basis), basis)

    for current_degree in range(1, degree + 1):
        next_basis = []
        for index in range(n_basis + degree - current_degree):
            left_den = knots[index + current_degree] - knots[index]
            right_den = knots[index + current_degree + 1] - knots[index + 1]
            left_term = torch.zeros_like(flat[:, 0])
            right_term = torch.zeros_like(flat[:, 0])
            if float(abs(left_den).item()) > 0.0:
                left_term = ((flat[:, 0] - knots[index]) / left_den) * basis[:, index]
            if float(abs(right_den).item()) > 0.0:
                right_term = ((knots[index + current_degree + 1] - flat[:, 0]) / right_den) * basis[:, index + 1]
            next_basis.append(left_term + right_term)
        basis = torch.stack(next_basis, dim=1)

    basis = basis[:, :n_basis]
    denom = basis.sum(dim=1, keepdim=True).clamp_min(torch.finfo(basis.dtype).eps)
    basis = basis / denom
    return basis.reshape(*x.shape, n_basis)
