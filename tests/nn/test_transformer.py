import torch
import pytest
from tests.constants import TOLERANCES, LOGM2_MEAN_STD, REPS, LFRAMES_PREDICTOR
from tests.helpers import sample_particle
from torch_geometric.utils import dense_to_sparse

from tensorframes.nn.transformer import TFTransformer
from tensorframes.reps.tensorreps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.utils.transforms import rand_lorentz
from tensorframes.lframes.lframes import InverseLFrames


@pytest.mark.parametrize("LFramesPredictor", LFRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("num_blocks", [0, 1, 2])
@pytest.mark.parametrize("attn_reps", REPS)
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
@pytest.mark.parametrize("spurion_strategy", [None])
def test_transformer_invariance_equivariance(
    LFramesPredictor,
    batch_dims,
    num_heads,
    num_blocks,
    logm2_std,
    logm2_mean,
    attn_reps,
    spurion_strategy,
):
    print(logm2_std, logm2_mean)
    dtype = torch.float64

    assert len(batch_dims) == 1
    predictor = LFramesPredictor(
        hidden_channels=16,
        num_layers=1,
        in_nodes=0,
        spurion_strategy=spurion_strategy,
    ).to(dtype=dtype)
    spurions = torch.zeros((0, 4), dtype=torch.long)
    edge_index = dense_to_sparse(torch.ones(batch_dims[0], batch_dims[0]))[0]
    scalars = torch.zeros(*batch_dims, 0, dtype=dtype)
    call_predictor = lambda fm: predictor(fm, scalars, edge_index, spurions)

    # define edgeconv
    in_reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(in_reps))
    net = TFTransformer(
        in_reps=in_reps,
        attn_reps=attn_reps,
        out_reps=in_reps,
        num_blocks=num_blocks,
        num_heads=num_heads,
    ).to(dtype=dtype)

    # get global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    lframes = call_predictor(fm)
    fm_local = trafo(fm, lframes)

    # global - edgeconv
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    lframes_transformed = call_predictor(fm_transformed)
    fm_tr_local = trafo(fm_transformed, lframes_transformed)
    fm_tr_prime_local = net(fm_tr_local, lframes_transformed)
    # back to global frame
    fm_tr_prime_global = trafo(fm_tr_prime_local, InverseLFrames(lframes_transformed))

    # edgeconv - global
    fm_prime_local = net(fm_local, lframes)
    # back to global
    fm_prime_global = trafo(fm_prime_local, InverseLFrames(lframes))
    fm_prime_tr_global = torch.einsum("...ij,...j->...i", random, fm_prime_global)

    # test equivariance of outputs
    torch.testing.assert_close(fm_tr_prime_global, fm_prime_tr_global, **TOLERANCES)
