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

    def forward(self, fourmomenta, scalars, edge_index, batch, return_tracker=False):
        vecs = super().forward(fourmomenta, scalars, edge_index)
        vecs = vecs.to(dtype=torch.float64)
        vecs = [vecs[..., i, :] for i in range(self.n_vectors)]

        trafo, frac_lightlike, frac_coplanar = orthogonal_trafo(
            vecs, **self.ortho_kwargs, return_frac=True
        )

        tracker = {"frac_lightlike": frac_lightlike, "frac_coplanar": frac_coplanar}
        lframes = LFrames(trafo.to(dtype=scalars.dtype))
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

    def forward(self, fourmomenta, scalars, edge_index, batch, return_tracker=False):
        references = super().forward(fourmomenta, scalars, edge_index)
        references = references.to(dtype=torch.float64)
        references = [references[..., i, :] for i in range(self.n_vectors)]
        fourmomenta = fourmomenta.to(dtype=torch.float64)

        trafo, frac_collinear = restframe_equivariant(
            fourmomenta,
            references,
            **self.ortho_kwargs,
            return_frac=True,
        )
        tracker = {"frac_collinear": frac_collinear}
        lframes = LFrames(trafo.to(dtype=scalars.dtype))
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

    def forward(self, fourmomenta, scalars, edge_index, batch, return_tracker=False):
        vecs = super().forward(fourmomenta, scalars, edge_index)
        vecs = vecs.to(dtype=torch.float64)
        fourmomenta = vecs[..., 0, :]
        references = [vecs[..., i, :] for i in range(1, self.n_vectors)]

        trafo, frac_collinear = restframe_equivariant(
            fourmomenta,
            references,
            **self.ortho_kwargs,
            return_frac=True,
        )
        tracker = {"frac_collinear": frac_collinear}
        lframes = LFrames(trafo.to(dtype=scalars.dtype))
        return (lframes, tracker) if return_tracker else lframes
