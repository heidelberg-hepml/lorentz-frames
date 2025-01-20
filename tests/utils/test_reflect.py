import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS
from tests.helpers import lorentz_test

from tensorframes.utils.reflect import (
    reflect_list,
)
from tensorframes.utils.transforms import rand_lorentz
from tensorframes.utils.lorentz import lorentz_metric


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("n_reflections", range(1, 5))
def test_lorentz(batch_dims, n_reflections):
    dtype = torch.float64  # some tests require higher precision

    vs = [torch.randn(*batch_dims, 4, dtype=dtype) for _ in range(n_reflections)]
    trafo = reflect_list(vs)

    lorentz_test(trafo, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("n_reflections", range(1, 5))
def test_equivariance(batch_dims, n_reflections):
    dtype = torch.float64  # some tests require higher precision

    vs = [torch.randn(*batch_dims, 4, dtype=dtype) for _ in range(n_reflections)]
    trafo = reflect_list(vs)

    random = rand_lorentz(batch_dims, dtype=dtype)
    metric = lorentz_metric(batch_dims, dtype=dtype)
    random_inv = torch.einsum(
        "...ij,...jk,...kl->...il", metric, random.transpose(-1, -2), metric
    )

    vs_prime = [torch.einsum("...ij,...j->...i", random, v) for v in vs]
    trafo_prime = reflect_list(vs_prime)

    # test that trafo transforms as
    # trafo_prime = random * trafo * random_inv
    trafo_prime_expected = torch.einsum(
        "...ij,...jk,...kl->...il", random, trafo, random_inv
    )
    torch.testing.assert_close(trafo_prime, trafo_prime_expected, **TOLERANCES)
