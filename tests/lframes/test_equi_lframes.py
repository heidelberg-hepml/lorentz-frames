import torch
import pytest
from torch_geometric.utils import dense_to_sparse
from tests.constants import TOLERANCES, LOGM2_MEAN, LOGM2_STD
from tests.helpers import sample_vector, lorentz_test

from tensorframes.reps import TensorReps
from tensorframes.lframes.equi_lframes import (
    RestLFrames,
    CrossLearnedLFrames,
    ReflectLearnedLFrames,
    MatrixExpLearnedLFrames,
    pseudo_trafo,
)
from tensorframes.utils.transforms import rand_lorentz
from tensorframes.lframes.lframes import LFrames


@pytest.mark.parametrize(
    "LFramesPredictor",
    [CrossLearnedLFrames, ReflectLearnedLFrames, MatrixExpLearnedLFrames],
)  # RestLFrames dont work yet - have to understand them better
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("logm2_std", LOGM2_STD)
@pytest.mark.parametrize(
    "logm2_mean", [-3]
)  # CrossLearnedLFrames fails for larger values
def test_lframes_transformation(LFramesPredictor, batch_dims, logm2_std, logm2_mean):
    dtype = torch.float64  # required to consistently pass tests

    # preparations
    if LFramesPredictor == RestLFrames:
        predictor = LFramesPredictor()
        call_predictor = lambda fm: predictor(fm)
    elif LFramesPredictor in [
        CrossLearnedLFrames,
        ReflectLearnedLFrames,
        MatrixExpLearnedLFrames,
    ]:
        assert len(batch_dims) == 1
        predictor = LFramesPredictor(hidden_channels=16, num_layers=1, in_nodes=0).to(
            dtype=dtype
        )
        batch = torch.zeros(batch_dims, dtype=torch.long)
        edge_index = dense_to_sparse(torch.ones(batch_dims[0], batch_dims[0]))[0]
        scalars = torch.zeros(*batch_dims, 0, dtype=dtype)
        call_predictor = lambda fm: predictor(fm, scalars, edge_index, batch)

    # sample Lorentz vectors
    fm = sample_vector(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # lframes for un-transformed fm
    lframes = call_predictor(fm)

    # random global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # lframes for transformed fm
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    lframes_prime = call_predictor(fm_prime)

    # check that lframes transform correctly
    # expect lframes_prime = lframes * random^-1
    inv_random = LFrames(random).inv
    lframes_prime_expected = torch.einsum(
        "...ij,...jk->...ik", lframes.matrices, inv_random
    )
    torch.testing.assert_close(
        lframes_prime_expected, lframes_prime.matrices, **TOLERANCES
    )


# TODO: Modify pseudo_trafo to make the lorentz_test lines pass for ReflectLearnedLFrames, MatrixExpLearnedLFrames
@pytest.mark.parametrize(
    "LFramesPredictor",
    [RestLFrames, CrossLearnedLFrames, ReflectLearnedLFrames, MatrixExpLearnedLFrames],
)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("logm2_std", LOGM2_STD)
@pytest.mark.parametrize(
    "logm2_mean", [-3]
)  # CrossLearnedLFrames fails for larger values
def test_feature_invariance(LFramesPredictor, batch_dims, logm2_std, logm2_mean):
    dtype = torch.float64  # required to consistently pass tests

    # preparations
    if LFramesPredictor == RestLFrames:
        predictor = LFramesPredictor()
        call_predictor = lambda fm: predictor(fm)
    elif LFramesPredictor in [
        CrossLearnedLFrames,
        ReflectLearnedLFrames,
        MatrixExpLearnedLFrames,
    ]:
        assert len(batch_dims) == 1
        predictor = LFramesPredictor(hidden_channels=16, num_layers=1, in_nodes=0).to(
            dtype=dtype
        )
        batch = torch.zeros(batch_dims, dtype=torch.long)
        edge_index = dense_to_sparse(torch.ones(batch_dims[0], batch_dims[0]))[0]
        scalars = torch.zeros(*batch_dims, 0, dtype=dtype)
        call_predictor = lambda fm: predictor(fm, scalars, edge_index, batch)

    reps = TensorReps("1x1n")
    trafo = TensorReps(reps).get_transform_class()

    # sample Lorentz vectors
    fm = sample_vector(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    if LFramesPredictor in [ReflectLearnedLFrames, MatrixExpLearnedLFrames]:
        torch.zeros(batch_dims, dtype=torch.long)
        pseudo = pseudo_trafo(fm, batch)
        # lorentz_test(pseudo, **TOLERANCES)

    # random global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # path 1: LFrames transform (+ random transform)
    lframes = call_predictor(fm)
    if LFramesPredictor in [RestLFrames, CrossLearnedLFrames]:
        lorentz_test(lframes.matrices, **TOLERANCES)
    fm_local = trafo(fm, lframes)

    # path 2: random transform + LFrames transform
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    lframes_prime = call_predictor(fm_prime)
    if LFramesPredictor in [RestLFrames, CrossLearnedLFrames]:
        lorentz_test(lframes_prime.matrices, **TOLERANCES)
    fm_local_prime = trafo(fm_prime, lframes_prime)

    # test that features are invariant
    torch.testing.assert_close(fm_local, fm_local_prime, **TOLERANCES)
