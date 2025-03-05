from tensorframes.lframes.lframes import LFrames
from tensorframes.lframes.nonequi_lframes import LFramesPredictor
from tensorframes.utils.restframe import restframe_equivariant
from tensorframes.utils.lorentz import lorentz_eye
from tensorframes.utils.orthogonalize import orthogonal_trafo


class LearnedLFrames(LFramesPredictor):
    """Abstract class for local LFrames constructed
    based on equivariantly predicted vectors"""

    def __init__(
        self,
        equivectors,
        n_vectors,
        ortho_kwargs={},
    ):
        super().__init__()
        self.ortho_kwargs = ortho_kwargs
        self.equivectors = equivectors(n_vectors=n_vectors)

    def __repr__(self):
        classname = self.__class__.__name__
        method = self.ortho_kwargs["method"]
        string = f"{classname}(method={method})"
        return string


class LearnedOrthogonalLFrames(LearnedLFrames):
    """
    Local frames from an orthonormal set of vectors
    constructed from equivariantly predicted vectors
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, n_vectors=3, **kwargs)

    def forward(self, fourmomenta, scalars, return_tracker=False, **kwargs):
        vecs = self.equivectors(fourmomenta, scalars, **kwargs)
        vecs = [vecs[..., i, :] for i in range(vecs.shape[-2])]

        trafo, reg_lightlike, reg_coplanar = orthogonal_trafo(
            vecs, **self.ortho_kwargs, return_reg=True
        )

        tracker = {"reg_lightlike": reg_lightlike, "reg_coplanar": reg_coplanar}
        lframes = LFrames(trafo)
        return (lframes, tracker) if return_tracker else lframes


class LearnedPolarDecompositionLFrames(LearnedLFrames):
    """Construct LFrames as learnable polar decomposition (boost+rotation)"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, n_vectors=3, **kwargs)

    def forward(self, fourmomenta, scalars, return_tracker=False, **kwargs):
        vecs = self.equivectors(fourmomenta, scalars, **kwargs)
        fourmomenta = vecs[..., 0, :]
        references = [vecs[..., i, :] for i in range(1, vecs.shape[-2])]

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
    """Rest frame transformation with learnable equivariant rotation.
    This is a special case of LearnedPolarDecompositionLFrames
    where the boost vector is the particle momentum."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, n_vectors=2, **kwargs)

    def forward(self, fourmomenta, scalars, return_tracker=False, **kwargs):
        references = self.equivectors(fourmomenta, scalars, **kwargs)
        references = [references[..., i, :] for i in range(references.shape[-2])]

        trafo, reg_collinear = restframe_equivariant(
            fourmomenta,
            references,
            **self.ortho_kwargs,
            return_reg=True,
        )
        tracker = {"reg_collinear": reg_collinear}
        lframes = LFrames(trafo)
        return (lframes, tracker) if return_tracker else lframes


class LearnedOrthogonal3DLFrames(LearnedLFrames):
    """O(3) special case of LearnedOrthogonalLFrames"""

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

    def forward(self, fourmomenta, scalars, return_tracker=False, **kwargs):
        references = self.equivectors(fourmomenta, scalars, **kwargs)
        fourmomenta = lorentz_eye(
            fourmomenta.shape[:-1], device=fourmomenta.device, dtype=fourmomenta.dtype
        )[
            ..., 0
        ]  # only difference compared to LearnedRestLFrames
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
