import torch
import pytest
from tests.constants import TOLERANCES, REPS

from tensorframes.lframes.lframes import LFrames, InverseLFrames, ChangeOfLFrames
from tensorframes.reps.tensorreps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.utils.transforms import rand_lorentz


@pytest.mark.parametrize("batch_dims", [[1000]])
@pytest.mark.parametrize("reps", REPS)
def test_equivariance(batch_dims, reps):
    dtype = torch.float64

    reps = TensorReps(reps)
    trafo = TensorRepsTransform(TensorReps(reps))

    transform = rand_lorentz(batch_dims, dtype=dtype)
    lframes = LFrames(transform)

    x = torch.randn(*batch_dims, reps.dim, dtype=dtype)

    # manual transform
    transform_direct = torch.einsum(
        "...ij,...jk->...ik", lframes.matrices, InverseLFrames(lframes).matrices
    )
    change_lframes1 = LFrames(transform_direct)
    x_prime1 = trafo(x, change_lframes1)
    torch.testing.assert_close(x, x_prime1, **TOLERANCES)

    # all-in-one transform
    change_lframes2 = ChangeOfLFrames(lframes, lframes)
    x_prime2 = trafo(x, change_lframes2)
    torch.testing.assert_close(x, x_prime2, **TOLERANCES)
