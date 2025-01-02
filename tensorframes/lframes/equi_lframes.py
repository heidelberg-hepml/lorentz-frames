import torch
from torch_geometric.utils import scatter

from tensorframes.lframes.lframes import LFrames
from tensorframes.lframes.nonequi_lframes import LFramesPredictor
from tensorframes.utils.transforms import restframe_transform
from tensorframes.nnhep.equivectors import EquivariantVectors
from tensorframes.utils.lorentz import (
    lorentz_inner,
    lorentz_squarednorm,
    lorentz_metric,
)
from tensorframes.utils.reflect import reflect_list
from tensorframes.utils.matrixexp import matrix_exponential


class RestLFrames(LFramesPredictor):
    """Local frames corresponding to the rest frames of the particles"""

    def __init__(self):
        super().__init__()

    def forward(self, fourmomenta):
        transform = restframe_transform(fourmomenta)
        return LFrames(transform)


class LearnedLFrames(LFramesPredictor):
    """Abstract class for local LFrames constructed
    based on equivariantly predicted vectors"""

    def __init__(
        self,
        n_vectors,
        hidden_channels,
        in_nodes,
        **mlp_kwargs,
    ):
        super().__init__()
        self.in_nodes = in_nodes
        self.equivectors = EquivariantVectors(
            n_vectors=n_vectors,
            in_nodes=in_nodes,
            in_edges=1,
            hidden_channels=hidden_channels,
            **mlp_kwargs,
        )

    def forward(self, fourmomenta, scalars, edge_index):
        assert scalars.shape[-1] == self.in_nodes
        edge_attr = lorentz_inner(
            fourmomenta[edge_index[1]], fourmomenta[edge_index[0]]
        ).unsqueeze(-1)
        vecs = self.equivectors(
            x=scalars,
            fm=fourmomenta,
            edge_attr=edge_attr,
            edge_index=edge_index,
        )
        return vecs


class ReflectLearnedLFrames(LearnedLFrames):
    """
    Local frames constructed using reflections
    based on equivariantly predicted vectors

    For the Lorentz group one requires n_vectors>=4
    to be able to represent any transformation using reflections,
    according to the Cartan Dieudonne theorem
    """

    def __init__(
        self,
        *args,
        n_vectors=4,
        **kwargs,
    ):
        self.n_vectors = n_vectors
        super().__init__(*args, n_vectors=self.n_vectors, **kwargs)

    def forward(self, fourmomenta, scalars, edge_index, batch):
        vecs = super().forward(fourmomenta, scalars, edge_index)
        vecs = [vecs[..., i, :] for i in range(self.n_vectors)]

        trafo = reflect_list(vecs)

        counter = pseudo_trafo(fourmomenta, batch)
        trafo = counter @ trafo
        return LFrames(trafo)


class MatrixExpLearnedLFrames(LearnedLFrames):
    """
    Local frames constructed using the matrix exponential
    of a generator created based on equivariantly predicted vectors
    """

    def __init__(
        self,
        stability_factor=20,
        *args,
        **kwargs,
    ):
        self.n_vectors = 2
        super().__init__(*args, n_vectors=self.n_vectors, **kwargs)

        # to avoid numerical instabilities from large values in matrix_exponential
        self.stability_factor = stability_factor

    def forward(self, fourmomenta, scalars, edge_index, batch):
        vecs = super().forward(fourmomenta, scalars, edge_index)
        vecs /= self.stability_factor
        vecs = [vecs[..., i, :] for i in range(self.n_vectors)]

        trafo = matrix_exponential(*vecs)

        counter = pseudo_trafo(fourmomenta, batch)
        trafo = counter @ trafo
        return LFrames(trafo)


def pseudo_trafo(fourmomenta, batch):
    """
    Construct a pseudo matrix P^a_mu
    with transformation behaviour P -> P L^-1 under a lorentz transform L.

    This is required to restore the correct transformation behaviour
    in LFrames approaches that start with properly constructed local
    lorentz transforms T^mu_nu and turns them into T^a_nu
    """
    assert len(fourmomenta.shape) == 2
    summed = scatter(fourmomenta, index=batch, dim=0, reduce="sum").index_select(
        0, batch
    )

    norm = lorentz_squarednorm(summed).sqrt().unsqueeze(-1)
    summed /= norm

    pseudo = summed.unsqueeze(-2).repeat(1, 4, 1)
    metric = lorentz_metric(
        fourmomenta.shape[:-1], device=fourmomenta.device, dtype=fourmomenta.dtype
    )
    pseudo = pseudo @ metric
    return pseudo
