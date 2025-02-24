import torch
import pytest
from torch_geometric.utils import dense_to_sparse
from tests.constants import TOLERANCES, LOGM2_MEAN_STD, LFRAMES_PREDICTOR
from tests.helpers import sample_particle, lorentz_test

from tensorframes.reps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.utils.transforms import rand_lorentz
from tensorframes.lframes.lframes import LFrames


@pytest.mark.parametrize("LFramesPredictor", LFRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
@pytest.mark.parametrize("symmetry_breaking", [None])
def test_lframes_transformation(
    LFramesPredictor, batch_dims, logm2_std, logm2_mean, symmetry_breaking
):
    dtype = torch.float64

    # preparations
    assert len(batch_dims) == 1
    predictor = LFramesPredictor(
        hidden_channels=16,
        num_layers=1,
        in_nodes=0,
        symmetry_breaking=symmetry_breaking,
    ).to(dtype=dtype)
    spurions = torch.zeros((0, 4), dtype=torch.long)
    edge_index = dense_to_sparse(torch.ones(batch_dims[0], batch_dims[0]))[0]
    scalars = torch.zeros(*batch_dims, 0, dtype=dtype)
    call_predictor = lambda fm: predictor(fm, scalars, edge_index, spurions)

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
@pytest.mark.parametrize("symmetry_breaking", [None])
def test_feature_invariance(
    LFramesPredictor, batch_dims, logm2_std, logm2_mean, symmetry_breaking
):
    dtype = torch.float64

    # preparations
    assert len(batch_dims) == 1
    predictor = LFramesPredictor(
        hidden_channels=16,
        num_layers=1,
        in_nodes=0,
        symmetry_breaking=symmetry_breaking,
    ).to(dtype=dtype)
    spurions = torch.zeros((0, 4), dtype=torch.long)
    edge_index = dense_to_sparse(torch.ones(batch_dims[0], batch_dims[0]))[0]
    scalars = torch.zeros(*batch_dims, 0, dtype=dtype)
    call_predictor = lambda fm: predictor(fm, scalars, edge_index, spurions=spurions)

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
