import torch
import pytest
from tests.constants import TOLERANCES, LOGM2_MEAN, LOGM2_STD
from tests.helpers import sample_vector
from torch_geometric.utils import dense_to_sparse

from tensorframes.nnhep.graphnet import EdgeConv
from tensorframes.lframes.equi_lframes import (
    RestLFrames,
    CrossLearnedLFrames,
    ReflectLearnedLFrames,
    MatrixExpLearnedLFrames,
)
from tensorframes.reps.tensorreps import TensorReps
from tensorframes.utils.transforms import rand_transform


@pytest.mark.parametrize("LFramesPredictor", [CrossLearnedLFrames])
@pytest.mark.parametrize("num_layers_mlp1", range(1, 3))
@pytest.mark.parametrize("num_layers_mlp2", range(1, 3))
@pytest.mark.parametrize("logm2_std", LOGM2_STD)
@pytest.mark.parametrize("logm2_mean", LOGM2_MEAN)
def test_edgeconv_feature_invariance(
    LFramesPredictor,
    num_layers_mlp1,
    num_layers_mlp2,
    logm2_std,
    logm2_mean,
    aggr="add",
):
    # test construction of the messages in EdgeConv by probing the feature invariance
    # preparations as in test_attention
    # only use 1 "jet"
    dtype = torch.float64  # is this needed?
    batch_dims = [10]
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
    reps = TensorReps("1x1n")
    trafo = TensorReps(reps).get_transform_class()
    edgeconv = EdgeConv(reps, num_layers_mlp1, num_layers_mlp2, aggr=aggr).to(dtype)

    # get a global transformation
    random = rand_transform([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_vector(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    lframes = call_predictor(fm)
    # move to local frame
    fm_local = trafo(fm, lframes)
    # transform with EdgeConv
    fm_local_prime = edgeconv(fm_local, lframes, edge_index)

    # now apply global first
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    lframes_transformed = call_predictor(fm_transformed)
    # move to local frame
    fm_transformed_local = trafo(fm_transformed, lframes_transformed)
    fm_tr_local_prime = edgeconv(fm_transformed_local, lframes_transformed, edge_index)

    torch.testing.assert_close(fm_local_prime, fm_tr_local_prime, **TOLERANCES)


@pytest.mark.parametrize("LFramesPredictor", [CrossLearnedLFrames])
@pytest.mark.parametrize("num_layers_mlp1", range(1, 2))
@pytest.mark.parametrize("num_layers_mlp2", range(0, 2))
@pytest.mark.parametrize("logm2_std", LOGM2_STD)
@pytest.mark.parametrize("logm2_mean", LOGM2_MEAN)
def test_edgeconv_equivariance(
    LFramesPredictor,
    num_layers_mlp1,
    num_layers_mlp2,
    logm2_std,
    logm2_mean,
    aggr="add",
):
    # test construction of the messages in EdgeConv by probing the equivariance
    # preparations as in test_attention
    # only use 1 "jet"
    dtype = torch.float64  # is this needed?
    batch_dims = [10]
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
    reps = TensorReps("1x1n")
    trafo = TensorReps(reps).get_transform_class()
    edgeconv = EdgeConv(reps, num_layers_mlp1, num_layers_mlp2, aggr=aggr).to(dtype)

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
    fm_tr_prime_local = edgeconv(fm_tr_local, lframes_transformed, edge_index)
    # back to global frame
    fm_tr_prime_global = trafo(fm_tr_prime_local, lframes_transformed.inverse_lframes())

    # edgeconv - global
    fm_prime_local = edgeconv(fm_local, lframes, edge_index)
    # back to global
    fm_prime_global = trafo(fm_prime_local, lframes.inverse_lframes())
    fm_prime_tr_global = torch.einsum("...ij,...j->...i", random, fm_prime_global)

    torch.testing.assert_close(fm_tr_prime_global, fm_prime_tr_global, **TOLERANCES)
