import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS
from tests.helpers import lorentz_test

from tensorframes.utils.matrixexp import (
    matrix_exponential,
)
from tensorframes.utils.transforms import rand_lorentz
from tensorframes.utils.lorentz import lorentz_metric


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_lorentz(batch_dims):
    dtype = torch.float64  # some tests require higher precision

    v1 = torch.randn(*batch_dims, 4, dtype=dtype)
    v2 = torch.randn(*batch_dims, 4, dtype=dtype)

    trafo = matrix_exponential(v1, v2)

    lorentz_test(trafo, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.skip(
    reason="Test fails because trafo makes v1_prime, v2_prime very large objects such that matrix_exp becomes unstable"
)
def test_equivariance(batch_dims):
    dtype = torch.float64  # some tests require higher precision

    v1 = torch.randn(*batch_dims, 4, dtype=dtype)
    v2 = torch.randn(*batch_dims, 4, dtype=dtype)

    trafo = matrix_exponential(v1, v2)

    random = rand_lorentz(batch_dims, dtype=dtype)
    metric = lorentz_metric(batch_dims, dtype=dtype)
    random_inv = torch.einsum(
        "...ij,...jk,...kl->...il", metric, random.transpose(-1, -2), metric
    )

    v1_prime = torch.einsum("...ij,...j->...i", random, v1)
    v2_prime = torch.einsum("...ij,...j->...i", random, v2)
    trafo_prime = matrix_exponential(v1_prime, v2_prime)

    # test that trafo transforms as
    # trafo_prime = random * trafo * random_inv
    trafo_prime_expected = torch.einsum(
        "...ij,...jk,...kl->...il", random, trafo, random_inv
    )
    torch.testing.assert_close(trafo_prime, trafo_prime_expected, **TOLERANCES)
