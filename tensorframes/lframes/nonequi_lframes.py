import torch
from tensorframes.lframes.lframes import LFrames
from tensorframes.utils.transforms import (
    rand_lorentz,
    rand_rotation,
    rand_xyrotation,
    rand_boost,
    rand_ztransform,
)


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
            is_global=True,
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

    def __init__(self, transform_type="lorentz", is_global=True, std_eta=0.5):
        super().__init__(is_global=is_global)
        self.is_global = is_global
        self.std_eta = std_eta
        self.transform_type = transform_type

    def transform(self, shape, device):
        if self.transform_type == "lorentz":
            return rand_lorentz(shape, std_eta=self.std_eta, device=device)
        elif self.transform_type == "rotation":
            return rand_rotation(shape, device=device)
        elif self.transform_type == "boost":
            return rand_boost(shape, std_eta=self.std_eta, device=device)
        elif self.transform_type == "xyrotation":
            return rand_xyrotation(shape, device=device)
        elif self.transform_type == "ztransform":
            return rand_ztransform(shape, std_eta=self.std_eta, device=device)
        else:
            raise ValueError(
                f"Transformation type {self.transform_type} not implemented"
            )

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        shape = (
            fourmomenta.shape[:-2] + (1,) if self.is_global else fourmomenta.shape[:-1]
        )
        matrix = self.transform(shape, device=fourmomenta.device)
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
        if self.transform_type in ["lorentz", "boost"]:
            string += f", std_eta={self.std_eta}"
        string += ")"
        return string
