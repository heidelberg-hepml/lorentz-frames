import torch
import pytest
from torch_geometric.utils import dense_to_sparse
from torch.nn import Linear
from tests.constants import TOLERANCES, LOGM2_MEAN_STD, REPS, LFRAMES_PREDICTOR
from tests.helpers import sample_particle

from tensorframes.reps.tensorreps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.nn.attention import InvariantParticleAttention
from tensorframes.lframes.lframes import InverseLFrames
from tensorframes.utils.transforms import rand_lorentz


@pytest.mark.parametrize("LFramesPredictor", LFRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("hidden_reps", REPS)
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
@pytest.mark.parametrize("symmetry_breaking", [None])
def test_invariance_equivariance(
    LFramesPredictor,
    batch_dims,
    hidden_reps,
    logm2_std,
    logm2_mean,
    symmetry_breaking,
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

    # preparations
    in_reps = TensorReps("1x1n")
    hidden_reps = TensorReps(hidden_reps)
    trafo = TensorRepsTransform(TensorReps(in_reps))
    attention = InvariantParticleAttention(hidden_reps).to(dtype=dtype)
    linear_in = Linear(in_reps.dim, 3 * hidden_reps.dim).to(dtype=dtype)
    linear_out = Linear(hidden_reps.dim, in_reps.dim).to(dtype=dtype)

    # random global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # path 1: LFrames transform + random transform
    lframes = call_predictor(fm)
    fm_local = trafo(fm, lframes)
    x_local = linear_in(fm_local).unsqueeze(0)
    q_local, k_local, v_local = x_local.chunk(3, dim=-1)
    x_local2 = attention(q_local, k_local, v_local, lframes).squeeze(0)
    fm_local = linear_out(x_local2)
    fm_global = trafo(fm_local, InverseLFrames(lframes))
    fm_global_prime = torch.einsum("...ij,...j->...i", random, fm_global)

    # path 2: random transform + LFrames transform
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    lframes_prime = call_predictor(fm_prime)
    fm_prime_local = trafo(fm_prime, lframes_prime)
    x_prime_local = linear_in(fm_prime_local).unsqueeze(0)
    q_prime_local, k_prime_local, v_prime_local = x_prime_local.chunk(3, dim=-1)
    x_prime_local2 = attention(
        q_prime_local, k_prime_local, v_prime_local, lframes_prime
    ).squeeze(0)
    fm_prime_local = linear_out(x_prime_local2)
    fm_prime_global = trafo(fm_prime_local, InverseLFrames(lframes_prime))

    # test feature invariance before the operation
    torch.testing.assert_close(x_local, x_prime_local, **TOLERANCES)

    # test feature invariance after the operation
    torch.testing.assert_close(x_local2, x_prime_local2, **TOLERANCES)

    # test equivariance of output
    torch.testing.assert_close(fm_prime_global, fm_global_prime, **TOLERANCES)
