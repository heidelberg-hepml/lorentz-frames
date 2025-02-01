import torch
import pytest
from torch_geometric.utils import dense_to_sparse
from tests.constants import TOLERANCES, LOGM2_MEAN, LOGM2_STD
from tests.helpers import sample_vector, sample_vector_realistic, lorentz_test

from tensorframes.utils.transforms import rand_lorentz
from tensorframes.lframes.lframes import LFrames
from tensorframes.lframes.equi_lframes_update import (
    MatrixExpLearnedLFrames,
    ReflectLearnedLFrames,
)


@pytest.mark.parametrize(
    "LFramesPredictor", [MatrixExpLearnedLFrames, ReflectLearnedLFrames]
)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("logm2_std", LOGM2_STD)
@pytest.mark.parametrize("logm2_mean", LOGM2_MEAN)
@pytest.mark.parametrize("vector_type", [sample_vector, sample_vector_realistic])
def test_update_lframes_transformation(
    LFramesPredictor, batch_dims, logm2_std, logm2_mean, vector_type
):
    dtype = torch.float64

    # preparations
    assert len(batch_dims) == 1
    predictor = LFramesPredictor(hidden_channels=16, num_layers=1, in_nodes=0).to(
        dtype=dtype
    )
    batch = torch.zeros(batch_dims, dtype=torch.long)
    edge_index = dense_to_sparse(torch.ones(batch_dims[0], batch_dims[0]))[0]
    scalars = torch.zeros(*batch_dims, 0, dtype=dtype)
    call_predictor = lambda fm: predictor(fm, scalars, edge_index, batch)

    # sample Lorentz vectors
    fm = vector_type(batch_dims, logm2_std, logm2_mean, dtype=dtype)

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
    # expect lframes_prime = random * lframes * random^-1
    inv_random = LFrames(random).inv
    lframes_prime_expected = torch.einsum(
        "...ij,...jk,...kl->...il", random, lframes.matrices, inv_random
    )
    torch.testing.assert_close(
        lframes_prime_expected, lframes_prime.matrices, **TOLERANCES
    )
