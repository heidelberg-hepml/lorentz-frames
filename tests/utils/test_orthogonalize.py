import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS

from tensorframes.utils.lorentz import (
    lorentz_inner,
    lorentz_squarednorm,
)
from tensorframes.utils.orthogonalize import (
    lorentz_cross,
    orthogonalize_cross,
    orthogonalize_cross_o3,
)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_lorentz_cross(batch_dims):
    # cross product of 3 random vectors
    v1 = torch.randn(batch_dims + [4])
    v2 = torch.randn(batch_dims + [4])
    v3 = torch.randn(batch_dims + [4])
    v4 = lorentz_cross(v1, v2, v3)

    # compute inner product of 4th vector with the first 3
    inner14 = lorentz_inner(v1, v4)
    inner24 = lorentz_inner(v2, v4)
    inner34 = lorentz_inner(v3, v4)

    # check that the inner products vanish
    zeros = torch.zeros_like(inner14)
    torch.testing.assert_close(inner14, zeros, **TOLERANCES)
    torch.testing.assert_close(inner24, zeros, **TOLERANCES)
    torch.testing.assert_close(inner34, zeros, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_orthogonalize(batch_dims):
    v1 = torch.randn(batch_dims + [4])
    v2 = torch.randn(batch_dims + [4])
    v3 = torch.randn(batch_dims + [4])

    orthogonal_vecs = orthogonalize_cross([v1, v2, v3])

    # test if there is only one time-like vector in each set of orthogonalized vectors
    norm = torch.stack([lorentz_squarednorm(v) for v in orthogonal_vecs], dim=-1)
    pos_norm = norm > 0
    torch.testing.assert_close(
        torch.sum(pos_norm, dim=-1),
        torch.ones(batch_dims).to(torch.int64),
        atol=0,
        rtol=0,
    )

    for i1, v1 in enumerate(orthogonal_vecs):
        for i2, v2 in enumerate(orthogonal_vecs):
            inner = lorentz_inner(v1, v2)
            target = torch.ones_like(inner) if i1 == i2 else torch.zeros_like(inner)
            torch.testing.assert_close(inner.abs(), target, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [[10000]])
@pytest.mark.parametrize("eps", [1e-10, 1e-5, 1e-2])
def test_orthogonalize_collinear(batch_dims, eps):
    # test for collinear (and also coplanar) vectors
    v1 = torch.randn(batch_dims + [4])
    v2 = torch.randn(batch_dims + [4])
    v3 = v1.clone() + eps * torch.randn(batch_dims + [4])

    orthogonal_vecs = orthogonalize_cross([v1, v2, v3])

    for i1, v1 in enumerate(orthogonal_vecs):
        for i2, v2 in enumerate(orthogonal_vecs):
            inner = lorentz_inner(v1, v2)
            target = torch.ones_like(inner) if i1 == i2 else torch.zeros_like(inner)
            torch.testing.assert_close(inner.abs(), target, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [[10000]])
@pytest.mark.parametrize("eps", [1e-10, 1e-5, 1e-2])
def test_orthogonalize_lightlike(batch_dims, eps):
    # test for a lightlike vector
    temp = torch.randn(batch_dims + [3])
    temp2 = torch.linalg.norm(temp, dim=-1, keepdim=True) + eps * torch.randn(
        batch_dims + [1]
    )
    v1 = torch.cat((temp2, temp), dim=-1)
    v2 = torch.randn(batch_dims + [4])
    v3 = torch.randn(batch_dims + [4])

    orthogonal_vecs = orthogonalize_cross([v1, v2, v3])

    for i1, v1 in enumerate(orthogonal_vecs):
        for i2, v2 in enumerate(orthogonal_vecs):
            inner = lorentz_inner(v1, v2)
            target = torch.ones_like(inner) if i1 == i2 else torch.zeros_like(inner)
            torch.testing.assert_close(inner.abs(), target, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_orthogonalize_o3(batch_dims):
    v1 = torch.randn(batch_dims + [3])
    v2 = torch.randn(batch_dims + [3])

    orthogonal_vecs = orthogonalize_cross_o3([v1, v2])

    for i1, v1 in enumerate(orthogonal_vecs):
        for i2, v2 in enumerate(orthogonal_vecs):
            inner = (v1 * v2).sum(dim=-1)
            target = torch.ones_like(inner) if i1 == i2 else torch.zeros_like(inner)
            torch.testing.assert_close(inner, target, **TOLERANCES)
