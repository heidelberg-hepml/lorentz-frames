import torch
import pytest
from tests.constants import TOLERANCES

from tensorframes.reps import TensorReps
from tensorframes.lframes.equi_lframes import RestLFrames
from tensorframes.utils.transforms import rand_transform


@pytest.mark.parametrize("LFramesPredictor", [RestLFrames])
@pytest.mark.parametrize("shape", [[10000]])
@pytest.mark.parametrize("logm2_std", [0.1, 1, 2])
@pytest.mark.parametrize("logm2_mean", [-3, 0, 3])
def test_equivariance(LFramesPredictor, shape, logm2_std, logm2_mean):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64  # required to consistently pass tests

    # preparations
    predictor = LFramesPredictor()
    reps = TensorReps("1x1n")
    trafo = TensorReps(reps).get_transform_class()

    # sample Lorentz vectors
    logm2 = torch.randn(*shape, 1, device=device, dtype=dtype) * logm2_std + logm2_mean
    p3 = torch.randn(*shape, 3, device=device, dtype=dtype)
    E = torch.sqrt(logm2.exp() + (p3**2).sum(dim=-1, keepdim=True))
    fm = torch.cat([E, p3], dim=-1)

    # path 1: RestLFrames transform
    lframes = predictor(fm)
    fm_local = trafo(fm, lframes)

    # path 2: random transform + RestLFrames transform
    random = rand_transform(fm.shape[:-1], device=device, dtype=dtype)
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    lframes_prime = predictor(fm_prime)
    fm_local_prime = trafo(fm_prime, lframes_prime)

    # test equivariance condition
    torch.testing.assert_close(fm_local, fm_local_prime, **TOLERANCES)
