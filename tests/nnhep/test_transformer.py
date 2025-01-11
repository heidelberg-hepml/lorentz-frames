import torch
import pytest
from tests.constants import TOLERANCES, LOGM2_MEAN, LOGM2_STD, REPS
from tests.helpers import sample_vector
from torch_geometric.utils import dense_to_sparse

from tensorframes.nnhep.transformer import TFTransformer
from tensorframes.lframes.equi_lframes import (
    RestLFrames,
    CrossLearnedLFrames,
    ReflectLearnedLFrames,
    MatrixExpLearnedLFrames,
)
from tensorframes.reps.tensorreps import TensorReps
from tensorframes.utils.transforms import rand_transform


@pytest.mark.parametrize("LFramesPredictor", [CrossLearnedLFrames])
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("num_heads", [1])  # TODO: extend to multiple heads
@pytest.mark.parametrize("num_blocks", [0, 1, 2])
@pytest.mark.parametrize("logm2_std", LOGM2_STD)
@pytest.mark.parametrize("logm2_mean", LOGM2_MEAN)
@pytest.mark.parametrize("reps", REPS)
def test_transformer_invariance_equivariance(
    LFramesPredictor,
    batch_dims,
    num_heads,
    num_blocks,
    logm2_std,
    logm2_mean,
    reps,
):
    dtype = torch.float64  # is this needed?

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

    # define edgeconv
    in_reps = TensorReps("1x1n")
    trafo = TensorReps(in_reps).get_transform_class()
    net = TFTransformer(
        in_channels=in_reps.dim,
        hidden_channels=reps,
        out_channels=in_reps.dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
    ).to(dtype=dtype)

    # get global transformation
    random = rand_transform([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_vector(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    lframes = call_predictor(fm)
    fm_local = trafo(fm, lframes)

    # global - edgeconv
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    lframes_transformed = call_predictor(fm_transformed)
    fm_tr_local = trafo(fm_transformed, lframes_transformed)
    fm_tr_prime_local = net(fm_tr_local, lframes_transformed)
    # back to global frame
    fm_tr_prime_global = trafo(fm_tr_prime_local, lframes_transformed.inverse_lframes())

    # edgeconv - global
    fm_prime_local = net(fm_local, lframes)
    # back to global
    fm_prime_global = trafo(fm_prime_local, lframes.inverse_lframes())
    fm_prime_tr_global = torch.einsum("...ij,...j->...i", random, fm_prime_global)

    # test equivariance of outputs
    torch.testing.assert_close(fm_tr_prime_global, fm_prime_tr_global, **TOLERANCES)
