import torch
import pytest
from tests.constants import TOLERANCES, LOGM2_MEAN, LOGM2_STD, REPS, LFRAMES_PREDICTOR
from tests.helpers import sample_vector, sample_vector_realistic
from torch_geometric.utils import dense_to_sparse

from tensorframes.nn.graphnet import EdgeConv, TFGraphNet
from tensorframes.reps.tensorreps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.utils.transforms import rand_lorentz
from tensorframes.lframes.lframes import InverseLFrames


@pytest.mark.parametrize("LFramesPredictor", LFRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("num_layers_mlp1", range(1, 2))
@pytest.mark.parametrize("num_layers_mlp2", range(0, 2))
@pytest.mark.parametrize("logm2_std", LOGM2_STD)
@pytest.mark.parametrize("logm2_mean", LOGM2_MEAN)
@pytest.mark.parametrize("reps", REPS)
@pytest.mark.parametrize("vector_type", [sample_vector, sample_vector_realistic])
def test_edgeconv_invariance_equivariance(
    LFramesPredictor,
    batch_dims,
    num_layers_mlp1,
    num_layers_mlp2,
    logm2_std,
    logm2_mean,
    reps,
    vector_type,
):
    # test construction of the messages in EdgeConv by probing the equivariance
    # preparations as in test_attention
    # only use 1 "jet"
    dtype = torch.float64  # is this needed?

    edge_index = dense_to_sparse(torch.ones(batch_dims[0], batch_dims[0]))[0]

    assert len(batch_dims) == 1
    predictor = LFramesPredictor(hidden_channels=16, num_layers=1, in_nodes=0).to(
        dtype=dtype
    )
    batch = torch.zeros(batch_dims, dtype=torch.long)
    scalars = torch.zeros(*batch_dims, 0, dtype=dtype)
    call_predictor = lambda fm: predictor(fm, scalars, edge_index, batch)

    # define edgeconv
    in_reps = TensorReps("1x1n")
    hidden_reps = TensorReps(reps)
    trafo = TensorRepsTransform(TensorReps(in_reps))
    linear_in = torch.nn.Linear(in_reps.dim, hidden_reps.dim).to(dtype=dtype)
    linear_out = torch.nn.Linear(hidden_reps.dim, in_reps.dim).to(dtype=dtype)
    edgeconv = EdgeConv(hidden_reps, num_layers_mlp1, num_layers_mlp2).to(dtype)

    # get global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = vector_type(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    lframes = call_predictor(fm)
    fm_local = trafo(fm, lframes)

    # global - edgeconv
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    lframes_transformed = call_predictor(fm_transformed)
    fm_tr_local = trafo(fm_transformed, lframes_transformed)
    x_tr_local = linear_in(fm_tr_local)
    x_tr_prime_local = edgeconv(x_tr_local, lframes_transformed, edge_index)
    fm_tr_prime_local = linear_out(x_tr_prime_local)
    # back to global frame
    fm_tr_prime_global = trafo(fm_tr_prime_local, InverseLFrames(lframes_transformed))

    # edgeconv - global
    x_local = linear_in(fm_local)
    x_prime_local = edgeconv(x_local, lframes, edge_index)
    fm_prime_local = linear_out(x_prime_local)
    # back to global
    fm_prime_global = trafo(fm_prime_local, InverseLFrames(lframes))
    fm_prime_tr_global = torch.einsum("...ij,...j->...i", random, fm_prime_global)

    # test feature invariance before the operation
    torch.testing.assert_close(x_local, x_tr_local, **TOLERANCES)

    # test feature invariance after the operation
    torch.testing.assert_close(x_tr_prime_local, x_prime_local, **TOLERANCES)

    # test equivariance of outputs
    torch.testing.assert_close(fm_tr_prime_global, fm_prime_tr_global, **TOLERANCES)


@pytest.mark.parametrize("LFramesPredictor", LFRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("num_layers_mlp1", range(1, 2))
@pytest.mark.parametrize("num_layers_mlp2", range(0, 2))
@pytest.mark.parametrize("num_blocks", [0, 1, 2])
@pytest.mark.parametrize("logm2_std", LOGM2_STD)
@pytest.mark.parametrize("logm2_mean", LOGM2_MEAN)
@pytest.mark.parametrize("reps", REPS)
@pytest.mark.parametrize("vector_type", [sample_vector, sample_vector_realistic])
def test_graphnet_invariance_equivariance(
    LFramesPredictor,
    batch_dims,
    num_layers_mlp1,
    num_layers_mlp2,
    num_blocks,
    logm2_std,
    logm2_mean,
    reps,
    vector_type,
):
    # test construction of the messages in GraphNet by probing the equivariance
    # preparations as in test_attention
    # only use 1 "jet"
    dtype = torch.float64  # is this needed?

    edge_index = dense_to_sparse(torch.ones(batch_dims[0], batch_dims[0]))[0]

    assert len(batch_dims) == 1
    predictor = LFramesPredictor(hidden_channels=16, num_layers=1, in_nodes=0).to(
        dtype=dtype
    )
    batch = torch.zeros(batch_dims, dtype=torch.long)
    scalars = torch.zeros(*batch_dims, 0, dtype=dtype)
    call_predictor = lambda fm: predictor(fm, scalars, edge_index, batch)

    # define edgeconv
    in_reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(in_reps))
    graphnet = TFGraphNet(
        in_channels=in_reps.dim,
        hidden_channels=reps,
        num_classes=in_reps.dim,
        num_blocks=num_blocks,
        num_layers_mlp1=num_layers_mlp1,
        num_layers_mlp2=num_layers_mlp2,
    ).to(dtype=dtype)

    # get global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = vector_type(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    lframes = call_predictor(fm)
    fm_local = trafo(fm, lframes)

    # global - edgeconv
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    lframes_transformed = call_predictor(fm_transformed)
    fm_tr_local = trafo(fm_transformed, lframes_transformed)
    fm_tr_prime_local = graphnet(fm_tr_local, lframes_transformed, edge_index)
    # back to global frame
    fm_tr_prime_global = trafo(fm_tr_prime_local, InverseLFrames(lframes_transformed))

    # edgeconv - global
    fm_prime_local = graphnet(fm_local, lframes, edge_index)
    # back to global
    fm_prime_global = trafo(fm_prime_local, InverseLFrames(lframes))
    fm_prime_tr_global = torch.einsum("...ij,...j->...i", random, fm_prime_global)

    # test equivariance of outputs
    torch.testing.assert_close(fm_tr_prime_global, fm_prime_tr_global, **TOLERANCES)
