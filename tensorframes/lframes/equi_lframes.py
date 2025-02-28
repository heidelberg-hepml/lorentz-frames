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
        spurion_strategy=None,
        *args,
        ortho_kwargs={},
        **kwargs,
    ):
        """
        contructor

        Args:
            n_vectors: The number of vectors to predict, this is usually 3, when the last vector is derived per cross product of the other 3 or 4
            in_nodes: number of in_nodes for network prediction of the equivariant networks
            spurion_strategy: string None, "particle_append", "particle_add", "basis_triplet" indicating the type of symmetry breaking used

        """
        super().__init__()

        self.in_nodes = in_nodes
        self.spurion_strategy = spurion_strategy
        if spurion_strategy == "basis_triplet":
            # this 2 is a hyperparameter assuming that
            n_vectors = n_vectors - 2
            assert (
                n_vectors >= 0
            ), f"Need to predict at least 1 vector, using basis means using 2 spurions instead of predicting them."

        self.predicted_n_vectors = n_vectors
        self.ortho_kwargs = ortho_kwargs

        self.equivectors = EquivariantVectors(
            n_vectors=self.predicted_n_vectors,
            in_nodes=in_nodes,
            in_edges=1,
            *args,
            **kwargs,
        )

        # standardization parameters for edge attributes
        self.register_buffer("edge_inited", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("edge_mean", torch.zeros(0))
        self.register_buffer("edge_std", torch.ones(1))

    def forward(self, fourmomenta, scalars, edge_index, spurions=None, batch=None):
        """
        Args:
            fourmomenta: (batch, 4)
            scalars: scalar and tagging_features in frame (batch, n_scalar)
            edge_index: edges (2, edge_index)
            spurions: all spurions (n_spurions, 4)
        batch: torch.tensor of shape (N,)
        Returns:
            vecs: predicted and combined with spurions if basis symmetry breaking vectors (batch, num_vecs, 4)"""
        assert scalars.shape[-1] == self.in_nodes

        # calculate and standardize edge attributes
        assert fourmomenta.shape[1] == 4
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

        if self.spurion_strategy == "particle_add":
            assert (
                spurions.shape[0] <= self.predicted_n_vectors
            ), f"Only predict {self.predicted_n_vectors} vectors, can not add all {spurions.shape[0]} spurions."
            expanded_spurions = (
                torch.cat(
                    [
                        spurions,
                        torch.zeros(
                            (self.predicted_n_vectors - spurions.shape[0], 4),
                            device=spurions.device,
                        ),
                    ],
                    dim=0,
                )
                .repeat(fourmomenta.shape[0], 1, 1)
                .reshape(fourmomenta.shape[0], -1)
            )
        else:
            expanded_spurions = None

        # call networks
        vecs = self.equivectors(
            x=scalars,
            fm=fourmomenta,
            edge_attr=edge_attr,
            edge_index=edge_index,
            spurions=expanded_spurions,
            batch=batch,
        )

        if self.spurion_strategy == "basis_triplet":
            vecs = torch.cat([spurions.repeat(vecs.shape[0], 1, 1), vecs], dim=-2)
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
