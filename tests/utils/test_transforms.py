import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS, LOGM2_STD, LOGM2_MEAN
from tests.helpers import sample_vector

from tensorframes.utils.transforms import (
    rand_transform,
    rand_rotation,
    rand_phirotation,
    restframe_transform,
)
from tensorframes.utils.lorentz import lorentz_norm


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("n_range", [[1, 1], [3, 5]])
@pytest.mark.parametrize("std_eta", [0.1, 1, 2])
@pytest.mark.parametrize(
    "transform_type", [rand_transform, rand_rotation, rand_phirotation]
)
def test_rand(batch_dims, n_range, std_eta, transform_type):
    dtype = torch.float64  # some tests require higher precision

    # collect N different kinds of transformations
    kwargs = {
        "shape": batch_dims,
        "dtype": dtype,
    }
    if transform_type in [rand_transform, rand_rotation]:
        kwargs["n_range"] = n_range
    if transform_type in [rand_transform]:
        kwargs["std_eta"] = std_eta
    transform = transform_type(**kwargs)

    # test that the transformation matrix T is orthogonal
    # i.e. T^T * M * T = M with the metric M = diag(1, -1, -1, -1)
    metric = torch.diag(torch.tensor([1, -1, -1, -1], dtype=transform.dtype))
    metric = metric.view((1,) * len(batch_dims) + metric.shape).repeat(
        *batch_dims, 1, 1
    )
    test = torch.einsum(
        "...ij,...jk,...kl->...il", transform, metric, transform.transpose(-1, -2)
    )
    torch.testing.assert_close(test, metric, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("logm2_std", LOGM2_STD)
@pytest.mark.parametrize("logm2_mean", LOGM2_MEAN)
def test_restframe_transform(batch_dims, logm2_std, logm2_mean):
    dtype = torch.float32

    # sample Lorentz vectors
    fm = sample_vector(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # determine transformation into rest frame
    rest_trafo = restframe_transform(fm)
    fm_rest = torch.einsum("...ij,...j->...i", rest_trafo, fm)

    # check that the transformed fourmomenta are in the rest frame,
    # i.e. their spatial components vanish and the temporal component is the mass
    torch.testing.assert_close(
        fm_rest[..., 1:], torch.zeros_like(fm[..., 1:]), **TOLERANCES
    )
    torch.testing.assert_close(fm_rest[..., 0] ** 2, lorentz_norm(fm), **TOLERANCES)
