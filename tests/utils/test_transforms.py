import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS
from tests.helpers import lorentz_test

from tensorframes.utils.transforms import (
    rand_transform,
    rand_rotation,
    rand_phirotation,
)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("n_range", [[1, 1], [3, 5]])
@pytest.mark.parametrize("std_eta", [0.1, 1, 2])
@pytest.mark.parametrize(
    "transform_type", [rand_transform, rand_rotation, rand_phirotation]
)
def test_rand_lorentz(batch_dims, n_range, std_eta, transform_type):
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

    lorentz_test(transform, **TOLERANCES)
