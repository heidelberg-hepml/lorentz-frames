import torch
import pytest
from torch_geometric.utils import dense_to_sparse
from tests.constants import TOLERANCES, LOGM2_MEAN_STD, LFRAMES_PREDICTOR
from tests.helpers import sample_particle, lorentz_test, equivectors_builder

from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.transforms import rand_lorentz
from lloca.lframes.lframes import LFrames


@pytest.mark.parametrize("LFramesPredictor", LFRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_lframes_transformation(LFramesPredictor, batch_dims, logm2_std, logm2_mean):
    dtype = torch.float64

    # preparations
    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = LFramesPredictor(equivectors=equivectors).to(dtype=dtype)
    call_predictor = lambda fm: predictor(fm)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # lframes for un-transformed fm
    lframes = call_predictor(fm)
    lorentz_test(lframes.matrices, **TOLERANCES)

    # random global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # lframes for transformed fm
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    lframes_prime = call_predictor(fm_prime)
    lorentz_test(lframes_prime.matrices, **TOLERANCES)

    # check that lframes transform correctly
    # expect lframes_prime = lframes * random^-1
    inv_random = LFrames(random).inv
    lframes_prime_expected = torch.einsum(
        "...ij,...jk->...ik", lframes.matrices, inv_random
    )
    torch.testing.assert_close(
        lframes_prime_expected, lframes_prime.matrices, **TOLERANCES
    )


@pytest.mark.parametrize("LFramesPredictor", LFRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_feature_invariance(LFramesPredictor, batch_dims, logm2_std, logm2_mean):
    dtype = torch.float64

    # preparations
    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = LFramesPredictor(equivectors=equivectors).to(dtype=dtype)
    call_predictor = lambda fm: predictor(fm)

    reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(reps))

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # random global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # path 1: LFrames transform (+ random transform)
    lframes = call_predictor(fm)
    lorentz_test(lframes.matrices, **TOLERANCES)
    fm_local = trafo(fm, lframes)

    # path 2: random transform + LFrames transform
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    lframes_prime = call_predictor(fm_prime)
    lorentz_test(lframes_prime.matrices, **TOLERANCES)
    fm_local_prime = trafo(fm_prime, lframes_prime)

    # test that features are invariant
    torch.testing.assert_close(fm_local, fm_local_prime, **TOLERANCES)
