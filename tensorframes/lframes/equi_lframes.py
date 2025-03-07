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
        *args,
        ortho_kwargs={},
        **kwargs,
    ):
        """
        Args:
            n_vectors: The number of vectors to predict, this is usually 3, when the last vector is derived per cross product of the other 3 or 4
            in_nodes: number of in_nodes for network prediction of the equivariant networks

        """
        super().__init__()

        self.in_nodes = in_nodes

        self.ortho_kwargs = ortho_kwargs

        self.equivectors = EquivariantVectors(
            n_vectors=n_vectors,
            in_nodes=in_nodes,
            in_edges=1,
            *args,
            **kwargs,
        )

        # standardization parameters for edge attributes
        self.register_buffer("edge_inited", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("edge_mean", torch.zeros(0))
        self.register_buffer("edge_std", torch.ones(1))

    def forward(self, fourmomenta, scalars, edge_index, batch=None):
        """
        Parameters
        ----------
        fourmomenta: torch.tensor of shape (N, 4)
        scalars: torch.tensor of shape (N, C)
        edge_index: torch.tensor of shape (2, E)
        batch: torch.tensor of shape (N,)
        """
        assert scalars.shape[-1] == self.in_nodes

        # calculate and standardize edge attributes
        mij2 = lorentz_squarednorm(
            fourmomenta[edge_index[0]] + fourmomenta[edge_index[1]]
        ).unsqueeze(-1)
        edge_attr = mij2.clamp(min=1e-10).log()

        # standardization
        if not self.edge_inited:
            self.edge_mean = edge_attr.mean()
            self.edge_std = edge_attr.std().clamp(min=1e-5)
            self.edge_inited.fill_(True)
        edge_attr = (edge_attr - self.edge_mean) / self.edge_std

        # call networks
        vecs = self.equivectors(
            x=scalars,
            fm=fourmomenta,
            edge_attr=edge_attr,
            edge_index=edge_index,
            batch=batch,
        )
        return vecs

    def __repr__(self):
        classname = self.__class__.__name__
        method = self.ortho_kwargs["method"]
        string = f"{classname}(method={method})"
        return string


class OrthogonalLearnedLFrames(LearnedLFrames):
    """
    Local frames from an orthonormal set of vectors
    constructed from equivariantly predicted vectors
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.n_vectors = 3
        super().__init__(*args, n_vectors=self.n_vectors, **kwargs)

    def forward(self, fourmomenta, scalars, *args, return_tracker=False, **kwargs):
        vecs = super().forward(fourmomenta, scalars, *args, **kwargs)
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
        **kwargs,
    ):
        self.n_vectors = 2
        super().__init__(
            *args,
            n_vectors=self.n_vectors,
            **kwargs,
        )

    def forward(self, fourmomenta, scalars, *args, return_tracker=False, **kwargs):
        references = super().forward(fourmomenta, scalars, *args, **kwargs)

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

    def forward(self, fourmomenta, scalars, *args, return_tracker=False, **kwargs):
        vecs = super().forward(fourmomenta, scalars, *args, **kwargs)

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
