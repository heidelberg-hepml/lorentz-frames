import torch
from tensorframes.lframes.lframes import LFrames
from tensorframes.utils.transforms import (
    rand_lorentz,
    rand_rotation_uniform,
    rand_rotation_naive,
    rand_xyrotation,
    rand_ztransform,
    rand_general_lorentz,
)
from tensorframes.utils.lorentz import lorentz_eye
from tensorframes.utils.restframe import restframe_boost


class LFramesPredictor(torch.nn.Module):
    def __init__(self, is_global=False):
        super().__init__()
        self.is_global = is_global

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        raise NotImplementedError


class IdentityLFrames(LFramesPredictor):
    """Identity local frames, corresponding to non-equivariant networks"""

    def __init__(self):
        super().__init__(is_global=True)

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        lframes = LFrames(
            is_identity=True,
            device=fourmomenta.device,
            dtype=fourmomenta.dtype,
            shape=fourmomenta.shape[:-1],
        )

        return (lframes, {}) if return_tracker else lframes

    def __repr__(self):
        return "IdentityLFrames()"


class RandomLFrames(LFramesPredictor):
    """Randomly generates a local frame for the whole batch,
    corresponding to data augmentation."""

    def __init__(
        self,
        transform_type="lorentz",
        is_global=True,
        std_eta=0.1,
        n_max_std_eta=3.0,
    ):
        super().__init__(is_global=is_global)
        self.is_global = is_global
        self.std_eta = std_eta
        self.transform_type = transform_type
        self.n_max_std_eta = n_max_std_eta

    def transform(self, shape, device, dtype):
        if self.transform_type == "lorentz":
            return rand_lorentz(
                shape,
                std_eta=self.std_eta,
                n_max_std_eta=self.n_max_std_eta,
                device=device,
                dtype=dtype,
            )
        elif self.transform_type == "rotation":
            return rand_rotation_uniform(shape, device=device, dtype=dtype)
        elif self.transform_type == "rotation_naive":
            return rand_rotation_naive(shape, device=device, dtype=dtype)
        elif self.transform_type == "xyrotation":
            return rand_xyrotation(shape, device=device, dtype=dtype)
        elif self.transform_type == "ztransform":
            return rand_ztransform(
                shape,
                std_eta=self.std_eta,
                n_max_std_eta=self.n_max_std_eta,
                device=device,
                dtype=dtype,
            )
        elif self.transform_type == "general_lorentz":
            return rand_general_lorentz(
                shape,
                std_eta=self.std_eta,
                n_max_std_eta=self.n_max_std_eta,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(
                f"Transformation type {self.transform_type} not implemented"
            )

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        if not self.training:
            lframes = LFrames(
                is_identity=True,
                shape=fourmomenta.shape[:-1],
                device=fourmomenta.device,
                dtype=fourmomenta.dtype,
            )
            return (lframes, {}) if return_tracker else lframes

        shape = (
            fourmomenta.shape[:-2] + (1,) if self.is_global else fourmomenta.shape[:-1]
        )
        matrix = self.transform(
            shape, device=fourmomenta.device, dtype=fourmomenta.dtype
        )
        matrix = matrix.expand(*fourmomenta.shape[:-1], 4, 4)

        lframes = LFrames(
            is_global=self.is_global,
            matrices=matrix,
            device=fourmomenta.device,
            dtype=fourmomenta.dtype,
        )
        return (lframes, {}) if return_tracker else lframes

    def __repr__(self):
        string = f"RandomLFrames(transform_type={self.transform_type}, is_global={self.is_global}"
        if self.transform_type in ["lorentz", "ztransform, general_lorentz"]:
            string += f", std_eta={self.std_eta}"
            string += f", n_max_std_eta={self.n_max_std_eta}"
        string += ")"
        return string


class COMRandomLFrames(RandomLFrames):
    """Modifies the forward function of RandomLFrames such that
    an additional boost is applied to the whole event.

    Only applicable to amplitude regression, the boost changes
    the reference frame to the center of mass of the incoming particles."""

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        if not self.training:
            lframes = LFrames(
                is_identity=True,
                shape=fourmomenta.shape[:-1],
                device=fourmomenta.device,
                dtype=fourmomenta.dtype,
            )
            return (lframes, {}) if return_tracker else lframes

        shape = (
            fourmomenta.shape[:-2] + (1,) if self.is_global else fourmomenta.shape[:-1]
        )
        matrix = self.transform(
            shape, device=fourmomenta.device, dtype=fourmomenta.dtype
        )

        # hardcoded for amplitudes
        reference_vector = fourmomenta[..., :2, :].sum(dim=-2, keepdims=True)
        reference_boost = restframe_boost(reference_vector)[..., :4, :4]

        matrix = torch.einsum("...ij,...jk->...ik", matrix, reference_boost)
        matrix = matrix.expand(*fourmomenta.shape[:-1], 4, 4)

        lframes = LFrames(
            is_global=self.is_global,
            matrices=matrix,
            device=fourmomenta.device,
            dtype=fourmomenta.dtype,
        )
        return (lframes, {}) if return_tracker else lframes

    def __repr__(self):
        string = f"COMLFrames(transform_type={self.transform_type}, is_global={self.is_global}"
        if self.transform_type in ["lorentz", "ztransform", "general_lorentz"]:
            string += f", std_eta={self.std_eta}"
            string += f", n_max_std_eta={self.n_max_std_eta}"
        string += ")"
        return string
