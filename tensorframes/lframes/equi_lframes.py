import torch
from torch_geometric.utils import scatter

from tensorframes.lframes.lframes import LFrames
from tensorframes.lframes.nonequi_lframes import LFramesPredictor
from tensorframes.utils.restframe import restframe_transform_v2
from tensorframes.nn.equivectors import EquivariantVectors
from tensorframes.utils.lorentz import (
    lorentz_inner,
    lorentz_squarednorm,
    lorentz_metric,
)
from tensorframes.utils.reflect import reflect_list
from tensorframes.utils.matrixexp import matrix_exponential
from tensorframes.utils.orthogonalize import (
    orthogonalize_cross,
    regularize_lightlike,
    regularize_collinear,
    regularize_coplanar,
)

from experiments.logger import LOGGER


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
        **kwargs,
    ):
        self.n_vectors = n_vectors
        super().__init__(*args, n_vectors=self.n_vectors, **kwargs)

        self.eps = eps
        self.regularize = regularize

    def forward(self, fourmomenta, scalars, edge_index, batch):
        vecs = super().forward(fourmomenta, scalars, edge_index)
        vecs = vecs.to(dtype=torch.float64)
        vecs = [vecs[..., i, :] for i in range(self.n_vectors)]
        vecs = torch.stack(vecs)

        if self.regularize:
            vecs = regularize_lightlike(vecs)
            vecs = regularize_collinear(vecs)
            vecs = regularize_coplanar(vecs)

        orthogonal_vecs = orthogonalize_cross(vecs, self.eps)
        trafo = torch.stack(orthogonal_vecs, dim=-2)

        # turn into transformation matrix
        metric = lorentz_metric(
            trafo.shape[:-2], device=trafo.device, dtype=trafo.dtype
        )
        trafo = trafo @ metric

        # sort vectors by norm -> first vector has >0 norm
        # this is necessary to get valid Lorentz transforms
        # see paper for why at most one vector can have >0 norm
        vecs = [trafo[..., i, :] for i in range(4)]
        norm = torch.stack([lorentz_squarednorm(v) for v in vecs], dim=-1)
        pos_norm = norm > 0
        num_pos_norm = pos_norm.sum(dim=-1)
        if len(torch.unique(num_pos_norm)) > 1:
            LOGGER.warning(
                f"Warning: find different number of norm>0 vectors: {torch.unique(num_pos_norm)}"
            )
        old_trafo = trafo.clone()
        # note: have to be careful with double masks ('mask' and 'pos_norm')
        mask = (num_pos_norm == 1).unsqueeze(-1).repeat(1, 4)
        trafo[..., 0, :] = torch.where(mask, old_trafo[pos_norm], old_trafo[..., 0, :])
        mask = mask.unsqueeze(-2).repeat(1, 3, 1)
        trafo[..., 1:, :] = torch.where(
            mask, old_trafo[~pos_norm].view(-1, 3, 4), old_trafo[..., 1:, :]
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
