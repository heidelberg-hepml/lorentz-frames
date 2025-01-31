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
    regularize_lightlike,
    regularize_collinear,
    regularize_coplanar,
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


@pytest.mark.parametrize("batch_dims", [[10000]])
def test_orthogonalize_cross(batch_dims):
    # check orthogonality after using the function
    dtype = torch.float64

    v1 = torch.randn(batch_dims + [4], dtype=dtype)
    v2 = torch.randn(batch_dims + [4], dtype=dtype)
    v3 = torch.randn(batch_dims + [4], dtype=dtype)

    orthogonal_vecs = orthogonalize_cross([v1, v2, v3])

    for i1, v1 in enumerate(orthogonal_vecs):
        for i2, v2 in enumerate(orthogonal_vecs):
            inner = lorentz_inner(v1, v2)
            target = torch.ones_like(inner) if i1 == i2 else torch.zeros_like(inner)
            torch.testing.assert_close(inner.abs(), target, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [[10000]])
def test_orthogonalize_timelike(batch_dims):
    # test if there is only one time-like vector in each set of orthogonalized vectors
    dtype = torch.float64

    v1 = torch.randn(batch_dims + [4], dtype=dtype)
    v2 = torch.randn(batch_dims + [4], dtype=dtype)
    v3 = torch.randn(batch_dims + [4], dtype=dtype)

    orthogonal_vecs = orthogonalize_cross([v1, v2, v3])

    norm = torch.stack([lorentz_squarednorm(v) for v in orthogonal_vecs], dim=-1)
    pos_norm = norm > 0
    torch.testing.assert_close(
        torch.sum(pos_norm, dim=-1),
        torch.ones(batch_dims).to(torch.int64),
        atol=0,
        rtol=0,
    )


"""
Circumnvent the collinear numerical issues by resampling 
the almost-collinear vector with a random one.
This is not the ideal solution but it is should allow us 
to train without assertion errors.
The exception option should probably go inside the orthogonalization function. 
The exception criterion is the deltaR between vectors.

- exception_eps defines the threshold applied to the criterion
- sampling_eps is a tunable parameter which modulates the deviation from the original vector.

With the current settings the percentage of modified vectors is:
- 100% for eps of 1.e-10
- ~99% for eps of 1.e-5
- <1% for eps of 1.e-2
"""


@pytest.mark.parametrize("exception", [True])
@pytest.mark.parametrize("exception_eps", [1e-4])
@pytest.mark.parametrize("sample_eps", [1, 1e-3])
@pytest.mark.parametrize("batch_dims", [[10000]])
@pytest.mark.parametrize("eps", [1e-10, 1e-5, 1e-2])
@pytest.mark.parametrize("rejection_regularize", [True, False])
def test_orthogonalize_collinear(
    batch_dims, eps, exception, exception_eps, sample_eps, rejection_regularize
):
    dtype = torch.float64

    # test for collinear (and also coplanar) vectors
    v1 = torch.randn(batch_dims + [4], dtype=dtype)
    v2 = torch.randn(batch_dims + [4], dtype=dtype)
    v3 = v1.clone() + eps * torch.randn(batch_dims + [4], dtype=dtype)
    if rejection_regularize:
        v4 = torch.randn(batch_dims + [4], dtype=dtype)
        v5 = torch.randn(batch_dims + [4], dtype=dtype)
        v6 = torch.randn(batch_dims + [4], dtype=dtype)
        vs = torch.stack([v1, v2, v3, v4, v5, v6])
    else:
        vs = torch.stack([v1, v2, v3])

    if exception:
        vs = regularize_collinear(
            vs,
            exception_eps=exception_eps,
            sample_eps=sample_eps,
            rejection_regularize=rejection_regularize,
        )
    vs = vs[:3]

    orthogonal_vecs = orthogonalize_cross(vs)

    for i1, v1 in enumerate(orthogonal_vecs):
        for i2, v2 in enumerate(orthogonal_vecs):
            inner = lorentz_inner(v1, v2)
            target = torch.ones_like(inner) if i1 == i2 else torch.zeros_like(inner)
            torch.testing.assert_close(inner.abs(), target, **TOLERANCES)


@pytest.mark.parametrize("exception", [True])
@pytest.mark.parametrize("exception_eps", [1e-5])
@pytest.mark.parametrize("sample_eps", [1, 1e-3])
@pytest.mark.parametrize("batch_dims", [[10000]])
@pytest.mark.parametrize("eps", [1e-10, 1e-5, 1e-2])
@pytest.mark.parametrize("rejection_regularize", [True, False])
def test_orthogonalize_collinear_v2(
    batch_dims, eps, exception, exception_eps, sample_eps, rejection_regularize
):
    dtype = torch.float64

    # create collinear vectors
    v1 = torch.randn(batch_dims + [4], dtype=dtype)
    v2 = torch.randn(batch_dims + [4], dtype=dtype)
    v3 = v1.clone() + eps * torch.randn(batch_dims + [4], dtype=dtype)
    if rejection_regularize:
        v4 = torch.randn(batch_dims + [4], dtype=dtype)
        v5 = torch.randn(batch_dims + [4], dtype=dtype)
        v6 = torch.randn(batch_dims + [4], dtype=dtype)
        vs = torch.stack([v1, v2, v3, v4, v5, v6])
    else:
        vs = torch.stack([v1, v2, v3])

    if exception:
        # apply coplanar correction to ALL vectors
        vs = regularize_coplanar(
            vs,
            exception_eps=exception_eps,
            sample_eps=sample_eps,
            rejection_regularize=rejection_regularize,
        )
    vs = vs[:3]

    orthogonal_vecs = orthogonalize_cross(vs)

    for i1, v1 in enumerate(orthogonal_vecs):
        for i2, v2 in enumerate(orthogonal_vecs):
            inner = lorentz_inner(v1, v2)
            target = torch.ones_like(inner) if i1 == i2 else torch.zeros_like(inner)
            torch.testing.assert_close(inner.abs(), target, **TOLERANCES)


@pytest.mark.parametrize("exception", [True])
@pytest.mark.parametrize("exception_eps", [1e-6])
@pytest.mark.parametrize("sample_eps", [1, 1e-3])
@pytest.mark.parametrize("batch_dims", [[10000]])
@pytest.mark.parametrize("eps", [1e-10, 1e-5, 1e-2])
@pytest.mark.parametrize("rejection_regularize", [True, False])
def test_orthogonalize_coplanar(
    batch_dims, eps, exception, exception_eps, sample_eps, rejection_regularize
):
    dtype = torch.float64

    # test for collinear (and also coplanar) vectors
    v1 = torch.randn(batch_dims + [4], dtype=dtype)
    v2 = torch.randn(batch_dims + [4], dtype=dtype)
    v3 = v1.clone() + v2.clone() + eps * torch.randn(batch_dims + [4], dtype=dtype)
    if rejection_regularize:
        v4 = torch.randn(batch_dims + [4], dtype=dtype)
        v5 = torch.randn(batch_dims + [4], dtype=dtype)
        v6 = torch.randn(batch_dims + [4], dtype=dtype)
        vs = torch.stack([v1, v2, v3, v4, v5, v6])
    else:
        vs = torch.stack([v1, v2, v3])

    if exception:
        vs = regularize_coplanar(
            vs,
            exception_eps=exception_eps,
            sample_eps=sample_eps,
            rejection_regularize=rejection_regularize,
        )
    vs = vs[:3]

    orthogonal_vecs = orthogonalize_cross(vs)

    for i1, v1 in enumerate(orthogonal_vecs):
        for i2, v2 in enumerate(orthogonal_vecs):
            inner = lorentz_inner(v1, v2)
            target = torch.ones_like(inner) if i1 == i2 else torch.zeros_like(inner)
            torch.testing.assert_close(inner.abs(), target, **TOLERANCES)


"""
Circumnvent the lightlike numerical issues by resampling 
the almost-lightlike vector with a random one.
This is not the ideal solution but it is should allow us 
to train without assertion errors.
The exception option should probably go inside the orthogonalization function. 
The exception riterion is the norm of the vectors

- exception_eps defines the threshold applied to the criterion
- sampling_eps is a tunable parameter which modulates the deviation from the original vector.

With the current settings the percentage of modified vectors is:
- 100% for eps of 1.e-10
- <1% for eps of 1.e-5
- 0% for eps of 1.e-2
"""


@pytest.mark.parametrize("exception", [True])
@pytest.mark.parametrize("exception_eps", [1e-8])
@pytest.mark.parametrize("sample_eps", [1, 1e-3])
@pytest.mark.parametrize("batch_dims", [[100000]])
@pytest.mark.parametrize("eps", [1e-10, 1e-5, 1e-2])
@pytest.mark.parametrize("rejection_regularize", [True, False])
def test_orthogonalize_lightlike(
    batch_dims, eps, exception, exception_eps, sample_eps, rejection_regularize
):
    dtype = torch.float64

    # test for a lightlike vector
    temp = torch.randn(batch_dims + [3], dtype=dtype)
    temp2 = torch.linalg.norm(temp, dim=-1, keepdim=True) + eps * torch.randn(
        batch_dims + [1], dtype=dtype
    )
    v1 = torch.cat((temp2, temp), dim=-1)
    v2 = torch.randn(batch_dims + [4], dtype=dtype)
    v3 = torch.randn(batch_dims + [4], dtype=dtype)
    if rejection_regularize:
        v4 = torch.randn(batch_dims + [4], dtype=dtype)
        v5 = torch.randn(batch_dims + [4], dtype=dtype)
        v6 = torch.randn(batch_dims + [4], dtype=dtype)
        vs = torch.stack([v1, v2, v3, v4, v5, v6])
    else:
        vs = torch.stack([v1, v2, v3])

    if exception:
        vs = regularize_lightlike(
            vs,
            exception_eps=exception_eps,
            sample_eps=sample_eps,
            rejection_regularize=rejection_regularize,
        )
    vs = vs[:3]

    orthogonal_vecs = orthogonalize_cross(vs)

    for i1, v1 in enumerate(orthogonal_vecs):
        for i2, v2 in enumerate(orthogonal_vecs):
            inner = lorentz_inner(v1, v2)
            target = torch.ones_like(inner) if i1 == i2 else torch.zeros_like(inner)
            torch.testing.assert_close(inner.abs(), target, **TOLERANCES)
