import torch
from torch_geometric.utils import scatter

from tensorframes.lframes.lframes import LFrames
from tensorframes.lframes.nonequi_lframes import LFramesPredictor
from tensorframes.utils.restframe import restframe_transform_v2
from tensorframes.nn.equivectors import EquivariantVectors
from tensorframes.utils.lorentz import (
    lorentz_squarednorm,
    lorentz_metric,
)
from tensorframes.utils.reflect import reflect_list
from tensorframes.utils.matrixexp import matrix_exponential
from tensorframes.utils.orthogonalize import cross_trafo
from tensorframes.utils.gram_schmidt import gramschmidt_trafo


class RestLFrames(LFramesPredictor):
    """Local frames corresponding to the rest frames of the particles"""

    def __init__(self):
        super().__init__()

    def forward(self, fourmomenta):
        fm = fourmomenta.to(dtype=torch.float64)
        transform = restframe_transform_v2(fm)
        return LFrames(transform.to(dtype=fm.dtype))


class LearnedLFrames(LFramesPredictor):
    """Abstract class for local LFrames constructed
    based on equivariantly predicted vectors"""

    def __init__(
        self,
        n_vectors,
        in_nodes,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_nodes = in_nodes
        self.equivectors = EquivariantVectors(
            n_vectors=n_vectors,
            in_nodes=in_nodes,
            in_edges=1,
            *args,
            **kwargs,
        )

        # standardization parameters for edge attributes
        self.register_buffer("inv_inited", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("inv_mean", torch.zeros(1))
        self.register_buffer("inv_std", torch.ones(1))

    def forward(self, fourmomenta, scalars, edge_index):
        assert scalars.shape[-1] == self.in_nodes

        # calculate and standardize edge attributes
        mij2 = lorentz_squarednorm(
            fourmomenta[edge_index[0]] + fourmomenta[edge_index[1]]
        ).unsqueeze(-1)
        edge_attr = mij2.clamp(min=1e-5).log()
        if not self.inv_inited:
            self.inv_mean = edge_attr.mean()
            self.inv_std = edge_attr.std().clamp(min=1e-5)
        edge_attr = (edge_attr - self.inv_mean) / self.inv_std

        # call networks
        vecs = self.equivectors(
            x=scalars,
            fm=fourmomenta,
            edge_attr=edge_attr,
            edge_index=edge_index,
        )
        return vecs


class CrossLearnedLFrames(LearnedLFrames):
    """
    Local frames constructed using repeated cross products
    of on equivariantly predicted vectors
    """

    def __init__(
        self,
        *args,
        n_vectors=3,
        eps=1e-10,
        regularize=False,  # The current regularization breaks the feature invariance in the local frames. This has to be addressed
        rejection_regularize=False,
        **kwargs,
    ):
        self.n_vectors = n_vectors
        self.rejection_regularize = rejection_regularize
        if rejection_regularize:
            assert (
                regularize
            ), "For rejection regularize to work, regularize needs to be enabled"
            self.n_vectors += 6  # hyperparameter
        super().__init__(*args, n_vectors=self.n_vectors, **kwargs)

        self.eps = eps
        self.regularize = regularize

    def forward(self, fourmomenta, scalars, edge_index, batch):
        vecs = super().forward(fourmomenta, scalars, edge_index)
        vecs = vecs.to(dtype=torch.float64)
        vecs = [vecs[..., i, :] for i in range(self.n_vectors)]
        vecs = torch.stack(vecs)

        trafo = cross_trafo(
            vecs,
            eps=self.eps,
            regularize=self.regularize,
            rejection_regularize=self.rejection_regularize,
        )

        return LFrames(trafo.to(dtype=fourmomenta.dtype))


class GramSchmidtLearnedLFrames(LearnedLFrames):
    """
    Local frames constructed using repeated cross products
    of on equivariantly predicted vectors
    """

    def __init__(
        self,
        *args,
        n_vectors=3,
        eps=1e-10,
        regularize=False,  # The current regularization breaks the feature invariance in the local frames. This has to be addressed
        rejection_regularize=False,
        **kwargs,
    ):
        self.n_vectors = n_vectors
        self.rejection_regularize = rejection_regularize
        if rejection_regularize:
            assert (
                regularize
            ), "For rejection regularize to work, regularize needs to be enabled"
            self.n_vectors += 6  # hyperparameter
        super().__init__(*args, n_vectors=self.n_vectors, **kwargs)

        self.eps = eps
        self.regularize = regularize

    def forward(self, fourmomenta, scalars, edge_index, batch):
        vecs = super().forward(fourmomenta, scalars, edge_index)
        vecs = vecs.to(dtype=torch.float64)
        vecs = [vecs[..., i, :] for i in range(self.n_vectors)]
        vecs = torch.stack(vecs)

        trafo = gramschmidt_trafo(
            vecs,
            eps=self.eps,
            regularize=self.regularize,
            rejection_regularize=self.rejection_regularize,
        )

        return LFrames(trafo.to(dtype=fourmomenta.dtype))


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
        vecs = vecs.to(dtype=torch.float64)
        vecs = [vecs[..., i, :] for i in range(self.n_vectors)]

        trafo = reflect_list(vecs)

        counter = pseudo_trafo(fourmomenta, batch)
        trafo = counter @ trafo
        return LFrames(trafo.to(dtype=fourmomenta.dtype))


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
        vecs = vecs.to(dtype=torch.float64)
        vecs /= self.stability_factor
        vecs = [vecs[..., i, :] for i in range(self.n_vectors)]

        trafo = matrix_exponential(*vecs)

        counter = pseudo_trafo(fourmomenta, batch)
        trafo = counter @ trafo
        return LFrames(trafo.to(dtype=fourmomenta.dtype))


def pseudo_trafo(fourmomenta, batch):
    """
    Construct a pseudo matrix P^a_mu
    with transformation behaviour P -> P L^-1 under a lorentz transform L.

    This is required to restore the correct transformation behaviour
    in LFrames approaches that start with properly constructed local
    lorentz transforms T^mu_nu and turns them into T^a_nu

    TODO: pseudo is not a proper Lorentz transformation
    have to modify it to pass tests and turn the architecture equivariant
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
