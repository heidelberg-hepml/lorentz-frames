import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS

from lloca.utils.lorentz import lorentz_cross, lorentz_inner


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
