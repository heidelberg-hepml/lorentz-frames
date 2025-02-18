import torch

from tensorframes.lframes.lframes import LFrames
from tensorframes.lframes.nonequi_lframes import LFramesPredictor
from tensorframes.utils.restframe import restframe_equivariant
from tensorframes.nn.equivectors import EquivariantVectors
from tensorframes.utils.lorentz import lorentz_squarednorm
from tensorframes.utils.orthogonalize import orthogonal_trafo


class LearnedLFrames(LFramesPredictor):
    """Abstract class for local LFrames constructed
    based on equivariantly predicted vectors"""

    def __init__(
        self,
        n_vectors,
        in_nodes,
        spurion_lframes_replacements=None,
        *args,
        **kwargs,
    ):
        """
        contructor

        Args:
            n_vectors: The number of vectors to predict, this is usually 3, when the last vector is derived per cross product of the other 3 or 4
            in_nodes: number of in_nodes for network prediction of the equivariant networks
            spurion_lframes: number of spurions to replace some of the otherwise predicted vectors

        """
        super().__init__()

        self.in_nodes = in_nodes
        self.spurion_lframes_replacements = spurion_lframes_replacements
        if spurion_lframes_replacements is not None:
            n_vectors = n_vectors - spurion_lframes_replacements
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

    def forward(self, fourmomenta, scalars, edge_index, spurions):
        assert scalars.shape[-1] == self.in_nodes
        assert (
            self.spurion_lframes_replacements is None
            or self.spurion_lframes_replacements == spurions.shape[0]
        )
        if self.spurion_lframes_replacements is not None:
            assert spurions is not None

        # calculate and standardize edge attributes
        assert (
            fourmomenta.shape[1] == 4
        ), f"fourmomenta vector does not have 4 components"
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

        if self.spurion_lframes_replacements is not None:
            vecs = torch.cat([vecs, spurions.repeat(vecs.shape[0], 1, 1)], dim=-2)
        return vecs


class OrthogonalLearnedLFrames(LearnedLFrames):
    """
    Local frames from an orthonormal set of vectors
    constructed from equivariantly predicted vectors
    """

    def __init__(
        self,
        *args,
        ortho_kwargs={},
        **kwargs,
    ):
        self.n_vectors = 3
        self.ortho_kwargs = ortho_kwargs
        super().__init__(*args, n_vectors=self.n_vectors, **kwargs)

    def forward(
        self, fourmomenta, scalars, edge_index, batch, spurions=None, return_tracker=False
    ):
        vecs = super().forward(fourmomenta, scalars, edge_index, spurions)
        vecs = [vecs[..., i, :] for i in range(self.n_vectors)]

        trafo, reg_lightlike, reg_coplanar = orthogonal_trafo(
            vecs, **self.ortho_kwargs, return_reg=True
        )

        tracker = {"reg_lightlike": reg_lightlike, "reg_coplanar": reg_coplanar}
        lframes = LFrames(trafo)
        return (lframes, tracker) if return_tracker else lframes


class RestLFrames(LearnedLFrames):
    """Rest frame transformation with learnable aspect"""

    def __init__(
        self,
        *args,
        ortho_kwargs={},
        **kwargs,
    ):
        self.n_vectors = 2
        super().__init__(
            *args,
            n_vectors=self.n_vectors,
            **kwargs,
        )

        self.ortho_kwargs = ortho_kwargs

    def forward(
        self, fourmomenta, scalars, edge_index, batch, spurions=None, return_tracker=False
    ):
        references = super().forward(fourmomenta, scalars, edge_index, spurions)
        references = [references[..., i, :] for i in range(self.n_vectors)]

        trafo, reg_collinear = restframe_equivariant(
            fourmomenta,
            references,
            **self.ortho_kwargs,
            return_reg=True,
        )
        tracker = {"reg_collinear": reg_collinear}
        lframes = LFrames(trafo)
        return (lframes, tracker) if return_tracker else lframes


class LearnedRestLFrames(LearnedLFrames):
    """Rest frame transformation with learnable aspect"""

    def __init__(
        self,
        *args,
        ortho_kwargs={},
        **kwargs,
    ):
        self.n_vectors = 3
        super().__init__(
            *args,
            n_vectors=self.n_vectors,
            operation="single",
            nonlinearity="exp",
            **kwargs,
        )

        self.ortho_kwargs = ortho_kwargs

    def forward(
        self, fourmomenta, scalars, edge_index, batch, spurions=None, return_tracker=False
    ):
        vecs = super().forward(fourmomenta, scalars, edge_index, spurions)
        fourmomenta = vecs[..., 0, :]
        references = [vecs[..., i, :] for i in range(1, self.n_vectors)]

        trafo, reg_collinear = restframe_equivariant(
            fourmomenta,
            references,
            **self.ortho_kwargs,
            return_reg=True,
        )
        tracker = {"reg_collinear": reg_collinear}
        lframes = LFrames(trafo)
        return (lframes, tracker) if return_tracker else lframes
