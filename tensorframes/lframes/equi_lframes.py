import torch

from tensorframes.lframes.lframes import LFrames
from tensorframes.lframes.nonequi_lframes import LFramesPredictor
from tensorframes.utils.transforms import restframe_transform
from tensorframes.nnhep.equivectors import EquivariantVectors
from tensorframes.utils.lorentz import lorentz_inner
from tensorframes.utils.reflect import reflect_list


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

    def forward(self, *args, **kwargs):
        vecs = super().forward(*args, **kwargs)
        vecs = [vecs[..., i, :] for i in range(self.n_vectors)]

        trafo = reflect_list(vecs)
        return LFrames(trafo)
