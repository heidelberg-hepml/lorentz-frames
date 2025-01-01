import torch
import pytest
from tests.constants import STRICT_TOLERANCES, BATCH_DIMS
from tests.helpers import lorentz_test

from tensorframes.utils.matrixexp import (
    matrix_exponential,
)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_lorentz(batch_dims):
    dtype = torch.float64  # some tests require higher precision

    v1 = torch.randn(*batch_dims, 4, dtype=dtype)
    v2 = torch.randn(*batch_dims, 4, dtype=dtype)

    trafo = matrix_exponential(v1, v2)

    lorentz_test(trafo, **STRICT_TOLERANCES)
