import torch
from torch_geometric.utils import scatter

from tensorframes.utils.utils import get_batch_from_ptr
from tensorframes.lframes.lframes import LFrames
from tensorframes.lframes.nonequi_lframes import LFramesPredictor
from tensorframes.utils.restframe import restframe_equivariant
from tensorframes.utils.lorentz import lorentz_eye, lorentz_squarednorm
from tensorframes.utils.orthogonalize import orthogonal_trafo


class LearnedLFrames(LFramesPredictor):
    """Abstract class for local LFrames constructed
    based on equivariantly predicted vectors"""

    def __init__(
        self,
        equivectors,
        n_vectors,
        is_global=False,
        random=False,
        fix_params=False,
        ortho_kwargs={},
    ):
        """
        Parameters
        ----------
        equivectors: nn.Module
            Network that equivariantly predicts vectors
        n_vectors: int
            Number of vectors to predict
        is_global: bool
            If True, average the predicted vectors to construct a global frame
        random: bool
            If True, re-initialize the equivectors at each forward pass
            This is a fancy way of doing data augmentation
        fix_params: bool
            Like random, but without the resampling
        ortho_kwargs: dict
            Keyword arguments for orthogonalization
        """
        super().__init__()
        self.ortho_kwargs = ortho_kwargs
        self.equivectors = equivectors(n_vectors=n_vectors)
        self.is_global = is_global
        self.random = random
        if random or fix_params:
            self.equivectors.requires_grad_(False)

    def init_weights_or_not(self):
        if self.random and self.training:
            self.equivectors.apply(init_weights)

    def globalize_vecs_or_not(self, vecs, ptr):
        return average_event(vecs, ptr) if self.is_global else vecs

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

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        self.init_weights_or_not()
        vecs = self.equivectors(fourmomenta, scalars=scalars, ptr=ptr)
        vecs = self.globalize_vecs_or_not(vecs, ptr)
        vecs = [vecs[..., i, :] for i in range(vecs.shape[-2])]

        trafo, reg_lightlike, reg_coplanar = orthogonal_trafo(
            vecs, **self.ortho_kwargs, return_reg=True
        )

        tracker = {"reg_lightlike": reg_lightlike, "reg_coplanar": reg_coplanar}
        lframes = LFrames(trafo, is_global=self.is_global)
        return (lframes, tracker) if return_tracker else lframes


class LearnedPolarDecompositionLFrames(LearnedLFrames):
    """Construct LFrames as learnable polar decomposition (boost+rotation)"""

    def __init__(
        self,
        *args,
        gamma_max=None,
        gamma_hardness=None,
        **kwargs,
    ):
        super().__init__(*args, n_vectors=3, **kwargs)
        self.gamma_max = gamma_max
        self.gamma_hardness = gamma_hardness

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        self.init_weights_or_not()
        vecs = self.equivectors(fourmomenta, scalars=scalars, ptr=ptr)
        vecs = self.globalize_vecs_or_not(vecs, ptr)
        fourmomenta = vecs[..., 0, :]
        references = [vecs[..., i, :] for i in range(1, vecs.shape[-2])]
        fourmomenta, reg_gammamax = self._clamp_boost(fourmomenta)

        trafo, reg_collinear = restframe_equivariant(
            fourmomenta,
            references,
            **self.ortho_kwargs,
            return_reg=True,
        )
        tracker = {"reg_collinear": reg_collinear}
        if reg_gammamax is not None:
            tracker["reg_gammamax"] = reg_gammamax
        lframes = LFrames(trafo, is_global=self.is_global)
        return (lframes, tracker) if return_tracker else lframes

    def _clamp_boost(self, x):
        if self.gamma_max is None:
            return x, None

        else:
            # carefully clamp gamma to keep boosts under control
            mass = lorentz_squarednorm(x).clamp(min=0).sqrt().unsqueeze(-1)
            beta = x[..., 1:] / x[..., [0]].clamp(min=1e-10)
            gamma = x[..., [0]] / mass
            reg_gammamax = (gamma > self.gamma_max).sum().cpu()
            gamma_reg = soft_clamp_max(
                gamma, max=self.gamma_max, hardness=self.gamma_hardness
            )
            beta_scaling = (
                torch.sqrt(
                    torch.clamp(1 - 1 / gamma_reg.clamp(min=1e-10).square(), min=1e-10)
                )
                / (beta**2).sum(dim=-1, keepdim=True).clamp(min=1e-10).sqrt()
            )
            beta_reg = beta * beta_scaling
            x_reg = mass * torch.cat((gamma_reg, gamma_reg * beta_reg), dim=-1)
            return x_reg, reg_gammamax


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

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        self.init_weights_or_not()
        references = self.equivectors(fourmomenta, scalars=scalars, ptr=ptr)
        references = self.globalize_vecs_or_not(references, ptr)
        references = [references[..., i, :] for i in range(references.shape[-2])]

        trafo, reg_collinear = restframe_equivariant(
            fourmomenta,
            references,
            **self.ortho_kwargs,
            return_reg=True,
        )
        tracker = {"reg_collinear": reg_collinear}
        lframes = LFrames(trafo, is_global=self.is_global)
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

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        self.init_weights_or_not()
        references = self.equivectors(fourmomenta, scalars=scalars, ptr=ptr)
        references = self.globalize_vecs_or_not(references, ptr)
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
        lframes = LFrames(trafo, is_global=self.is_global)
        return (lframes, tracker) if return_tracker else lframes


def average_event(vecs, ptr=None):
    if ptr is None:
        vecs = vecs.mean(dim=-3, keepdim=True).expand_as(vecs)
    else:
        batch = get_batch_from_ptr(ptr)
        vecs = scatter(vecs, batch, dim=0, reduce="mean").index_select(0, batch)
    return vecs


def init_weights(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def soft_clamp_max(x, max, hardness=None):
    if hardness is None:
        # hard clamp
        return x.clamp(max=max)
    else:
        # soft clamp (better gradients)
        return max - torch.nn.functional.softplus(max - x, beta=hardness)
