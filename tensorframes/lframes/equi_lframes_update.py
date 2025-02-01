import torch

from tensorframes.lframes.lframes import LFrames
from tensorframes.lframes.equi_lframes import LearnedLFrames
from tensorframes.utils.reflect import reflect_list
from tensorframes.utils.matrixexp import matrix_exponential


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
        return LFrames(trafo.to(dtype=fourmomenta.dtype))
