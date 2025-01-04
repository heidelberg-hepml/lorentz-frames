import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS, LOGM2_STD, LOGM2_MEAN
from tests.helpers import sample_vector, lorentz_test

from tensorframes.utils.restframe import (
    restframe_transform,
    restframe_transform_2,
)
from tensorframes.utils.lorentz import lorentz_squarednorm


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "restframe_transform", [restframe_transform, restframe_transform_2]
)
@pytest.mark.parametrize("logm2_std", LOGM2_STD)
@pytest.mark.parametrize("logm2_mean", LOGM2_MEAN)
def test_restframe(batch_dims, restframe_transform, logm2_std, logm2_mean):
    dtype = torch.float64  # some tests require higher precision

    # sample Lorentz vectors
    fm = sample_vector(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # determine transformation into rest frame
    rest_trafo = restframe_transform(fm)
    fm_rest = torch.einsum("...ij,...j->...i", rest_trafo, fm)
    print(fm_rest[0])

    # check that the transformed fourmomenta are in the rest frame,
    # i.e. their spatial components vanish and the temporal component is the mass
    torch.testing.assert_close(
        fm_rest[..., 1:], torch.zeros_like(fm[..., 1:]), **TOLERANCES
    )
    torch.testing.assert_close(
        fm_rest[..., 0] ** 2, lorentz_squarednorm(fm), **TOLERANCES
    )

    lorentz_test(rest_trafo, **TOLERANCES)
