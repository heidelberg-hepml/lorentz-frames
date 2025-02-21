import torch
import pytest
from torch_geometric.utils import dense_to_sparse
from tests.constants import TOLERANCES, LOGM2_MEAN_STD
from tests.helpers import sample_particle

from tensorframes.nn.equivectors import EquivariantVectors
from tensorframes.utils.lorentz import lorentz_inner
from tensorframes.utils.transforms import rand_lorentz


@pytest.mark.parametrize("batch_dims", [[100]])
@pytest.mark.parametrize("jet_size", [10])
@pytest.mark.parametrize("n_vectors", range(1, 5))
@pytest.mark.parametrize("hidden_channels", [16])
@pytest.mark.parametrize("num_layers", [1])
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
@pytest.mark.parametrize("operation", ["diff", "add", "single"])
@pytest.mark.parametrize("nonlinearity", ["softplus", "exp", None])
def test_equivariance(
    batch_dims,
    jet_size,
    n_vectors,
    hidden_channels,
    num_layers,
    logm2_std,
    logm2_mean,
    operation,
    nonlinearity,
):
    assert len(batch_dims) == 1
    dtype = torch.float64

    # construct sparse tensors containing a set of equal-multiplicity jets
    ptr = torch.arange(0, (batch_dims[0] + 1) * jet_size, jet_size)
    diffs = torch.diff(ptr)
    edge_index = torch.cat(
        [
            dense_to_sparse(torch.ones(d, d, device=diffs.device))[0] + diffs[:i].sum()
            for i, d in enumerate(diffs)
        ],
        dim=-1,
    )

    # input to mlp: only edge attributes
    in_nodes = 0
    in_edges = 1
    calc_node_attr = lambda fm: torch.zeros(*fm.shape[:-1], 0, dtype=dtype)
    calc_edge_attr = lambda fm: lorentz_inner(
        fm[edge_index[1]], fm[edge_index[0]]
    ).unsqueeze(-1)
    equivectors = EquivariantVectors(
        n_vectors,
        in_nodes,
        in_edges,
        hidden_channels,
        num_layers,
        operation=operation,
        nonlinearity=nonlinearity,
    ).to(dtype=dtype)

    fm = sample_particle(
        batch_dims + [jet_size], logm2_std, logm2_mean, dtype=dtype
    ).flatten(0, 1)

    spurions = torch.zeros((fm.shape[0], n_vectors * 4), dtype=torch.long)

    # careful: same global transformation for each jet
    random = rand_lorentz(batch_dims, dtype=dtype)
    random = random.unsqueeze(1).repeat(1, jet_size, 1, 1).view(*fm.shape, 4)

    # path 1: global transform + predict vectors
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    edge_attr_prime = calc_edge_attr(fm_prime)
    node_attr_prime = calc_node_attr(fm_prime)
    vecs_prime1 = equivectors(
        node_attr_prime,
        fm_prime,
        edge_attr=edge_attr_prime,
        edge_index=edge_index,
        spurions=spurions,
    )

    # path 2: predict vectors + global transform
    edge_attr = calc_edge_attr(fm)
    node_attr = calc_node_attr(fm)
    vecs = equivectors(
        node_attr, fm, edge_attr=edge_attr, edge_index=edge_index, spurions=spurions
    )
    vecs_prime2 = torch.einsum("...ij,...kj->...ki", random, vecs)

    # test that edge attributes are invariant
    torch.testing.assert_close(edge_attr, edge_attr_prime, **TOLERANCES)

    # test that vectors are predicted equivariantly
    torch.testing.assert_close(vecs_prime1, vecs_prime2, **TOLERANCES)
