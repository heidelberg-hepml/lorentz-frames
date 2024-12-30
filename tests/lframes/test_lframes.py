import torch
import pytest
from tests.constants import TOLERANCES

from tensorframes.lframes import LFrames, ChangeOfLFrames
from tensorframes.reps.tensorreps import TensorReps
from tensorframes.utils.transforms import rand_transform


@pytest.mark.parametrize("shape", [[1000]])
@pytest.mark.parametrize("reps", ["10x0n+5x1n+2x2n"])
@pytest.mark.parametrize("logm2_std", [0.1])
@pytest.mark.parametrize("logm2_mean", [3])
def test_equivariance(shape, reps, logm2_std, logm2_mean):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    reps = TensorReps(reps)
    trafo = TensorReps(reps).get_transform_class()

    transform = rand_transform(shape, device=device, dtype=dtype)
    lframes = LFrames(transform)

    x = torch.randn(*shape, reps.dim, device=device, dtype=dtype)

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
