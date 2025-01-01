import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS
from tests.helpers import lorentz_test

from tensorframes.utils.reflect import (
    reflect_list,
)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("n_reflections", range(1, 5))
def test_lorentz(batch_dims, n_reflections):
    dtype = torch.float64  # some tests require higher precision

    vs = [torch.randn(*batch_dims, 4, dtype=dtype) for _ in range(n_reflections)]
    trafo = reflect_list(vs)

    lorentz_test(trafo, **TOLERANCES)
