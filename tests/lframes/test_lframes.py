import torch
import pytest
from tests.constants import TOLERANCES, REPS

from tensorframes.lframes import LFrames, ChangeOfLFrames
from tensorframes.reps.tensorreps import TensorReps
from tensorframes.utils.transforms import rand_transform


@pytest.mark.parametrize("batch_dims", [[1000]])
@pytest.mark.parametrize("reps", REPS)
def test_equivariance(batch_dims, reps):
    dtype = torch.float64

    reps = TensorReps(reps)
    trafo = TensorReps(reps).get_transform_class()

    transform = rand_transform(batch_dims, dtype=dtype)
    lframes = LFrames(transform)

    x = torch.randn(*batch_dims, reps.dim, dtype=dtype)

    # manual transform
    transform_direct = torch.einsum(
        "...ij,...jk->...ik", lframes.matrices, lframes.inverse_lframes().matrices
    )
    change_lframes1 = LFrames(transform_direct)
    x_prime1 = trafo(x, change_lframes1)
    torch.testing.assert_close(x, x_prime1, **TOLERANCES)

    # all-in-one transform
    change_lframes2 = ChangeOfLFrames(lframes, lframes)
    x_prime2 = trafo(x, change_lframes2)
    torch.testing.assert_close(x, x_prime2, **TOLERANCES)
