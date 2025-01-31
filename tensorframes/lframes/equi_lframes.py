import torch

from tensorframes.lframes.lframes import LFrames
from tensorframes.lframes.nonequi_lframes import LFramesPredictor
from tensorframes.utils.restframe import restframe_equivariant
from tensorframes.nn.equivectors import EquivariantVectors
from tensorframes.utils.lorentz import lorentz_squarednorm
from tensorframes.utils.orthogonalize import cross_trafo
from tensorframes.utils.gram_schmidt import gramschmidt_trafo


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

        # to keep track of regularized learned frames
        self.cumsum_lightlike = 0.0
        self.cumsum_coplanar = 0.0

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
        eps=1e-10,
        regularize=False,  # The current regularization breaks the feature invariance in the local frames. This has to be addressed
        rejection_regularize=False,
        **kwargs,
    ):
        self.n_vectors = 3
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

        trafo, n_light, n_space = cross_trafo(
            vecs,
            eps=self.eps,
            regularize=self.regularize,
            rejection_regularize=self.rejection_regularize,
        )

        self.cumsum_lightlike += n_light
        self.cumsum_coplanar += n_space

        return LFrames(trafo.to(dtype=fourmomenta.dtype))


class GramSchmidtLearnedLFrames(LearnedLFrames):
    """
    Local frames constructed using repeated cross products
    of on equivariantly predicted vectors
    """

    def __init__(
        self,
        *args,
        eps=1e-10,
        regularize=False,  # The current regularization breaks the feature invariance in the local frames. This has to be addressed
        rejection_regularize=False,
        regularize_coplanar_eps=1.0e-6,
        **kwargs,
    ):
        self.n_vectors = 3
        self.rejection_regularize = rejection_regularize
        if rejection_regularize:
            assert (
                regularize
            ), "For rejection regularize to work, regularize needs to be enabled"
            self.n_vectors += 6  # hyperparameter
        super().__init__(*args, n_vectors=self.n_vectors, **kwargs)

        self.eps = eps
        self.regularize_coplanar_eps = regularize_coplanar_eps
        self.regularize = regularize

    def forward(self, fourmomenta, scalars, edge_index, batch):
        vecs = super().forward(fourmomenta, scalars, edge_index)
        vecs = vecs.to(dtype=torch.float64)
        vecs = [vecs[..., i, :] for i in range(self.n_vectors)]
        vecs = torch.stack(vecs)

        trafo, n_light, n_space = gramschmidt_trafo(
            vecs,
            eps=self.eps,
            regularize=self.regularize,
            rejection_regularize=self.rejection_regularize,
            regularize_coplanar_eps=self.regularize_coplanar_eps,
        )

        self.cumsum_lightlike += n_light
        self.cumsum_coplanar += n_space

        return LFrames(trafo.to(dtype=fourmomenta.dtype))


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

    def forward(self, fourmomenta, scalars, edge_index, batch):
        references = super().forward(fourmomenta, scalars, edge_index)
        references = references.to(dtype=torch.float64)
        references = [references[..., i, :] for i in range(self.n_vectors)]
        fourmomenta = fourmomenta.to(dtype=torch.float64)

        trafo = restframe_equivariant(
            fourmomenta,
            references,
            **self.ortho_kwargs,
        )

        return LFrames(trafo.to(dtype=scalars.dtype))


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

    def forward(self, fourmomenta, scalars, edge_index, batch):
        vecs = super().forward(fourmomenta, scalars, edge_index)
        vecs = vecs.to(dtype=torch.float64)
        fourmomenta = vecs[..., 0, :]
        references = [vecs[..., i, :] for i in range(1, self.n_vectors)]

        trafo = restframe_equivariant(
            fourmomenta,
            references,
            **self.ortho_kwargs,
        )

        return LFrames(trafo.to(dtype=scalars.dtype))
