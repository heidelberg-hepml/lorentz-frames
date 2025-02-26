import torch
from tensorframes.lframes.lframes import LFrames
from tensorframes.utils.transforms import (
    rand_lorentz,
    rand_rotation,
    rand_phirotation,
    rand_boost,
)


class LFramesPredictor(torch.nn.Module):
    def __init__(self, is_global=False) -> None:
        super().__init__()
        self.is_global = is_global

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class IdentityLFrames(LFramesPredictor):
    """Identity local frames, corresponding to non-equivariant networks"""

    def __init__(self):
        super().__init__(is_global=True)

    def forward(self, fourmomenta, return_tracker=False):
        lframes = LFrames(
            is_global=True,
            is_identity=True,
            device=fourmomenta.device,
            dtype=fourmomenta.dtype,
            shape=fourmomenta.shape[:-1],
        )

        return (lframes, {}) if return_tracker else lframes


class RandomLFrames(LFramesPredictor):
    """Randomly generates a local frame for the whole batch,
    corresponding to data augmentation."""

    def __init__(self, std_eta=0.5):
        super().__init__(is_global=True)
        self.std_eta = std_eta

    def forward(self, fourmomenta, return_tracker=False):
        # general random transformation
        shape = fourmomenta.shape[:-2] + (1,)
        matrix = rand_lorentz(shape, std_eta=self.std_eta, device=fourmomenta.device)
        matrix = matrix.expand(*fourmomenta.shape[:-1], 4, 4)

        lframes = LFrames(
            is_global=True,
            matrices=matrix,
            device=fourmomenta.device,
            dtype=fourmomenta.dtype,
        )
        return (lframes, {}) if return_tracker else lframes


class RandomRotLFrames(LFramesPredictor):
    """Randomly generates a rotated local frame,
    corresponding to data augmentation."""

    def __init__(self, std_eta=1.0):
        super().__init__(is_global=True)

    def forward(self, fourmomenta, return_tracker=False):
        # random rotation
        shape = fourmomenta.shape[:-2] + (1,)
        matrix = rand_rotation(shape, device=fourmomenta.device)
        matrix = matrix.expand(*fourmomenta.shape[:-1], 4, 4)

        lframes = LFrames(
            is_global=True,
            matrices=matrix,
            device=fourmomenta.device,
            dtype=fourmomenta.dtype,
        )
        return (lframes, {}) if return_tracker else lframes


class RandomPhiLFrames(LFramesPredictor):
    """Randomly generates a phi-rotated local frame,
    corresponding to data augmentation with rotations around the z axis."""

    def __init__(self):
        super().__init__(is_global=True)

    def forward(self, fourmomenta, return_tracker=False):
        # random rotation around z axis
        shape = fourmomenta.shape[:-2] + (1,)
        matrix = rand_phirotation(shape, device=fourmomenta.device)
        matrix = matrix.expand(*fourmomenta.shape[:-1], 4, 4)

        lframes = LFrames(
            is_global=True,
            matrices=matrix,
            device=fourmomenta.device,
            dtype=fourmomenta.dtype,
        )
        return (lframes, {}) if return_tracker else lframes


class RandomBoostLFrames(LFramesPredictor):
    """Randomly generates a boosted local frame,
    corresponding to data augmentation."""

    def __init__(self, std_eta=0.5):
        super().__init__(is_global=True)
        self.std_eta = std_eta

    def forward(self, fourmomenta, return_tracker=False):
        # random boost
        shape = fourmomenta.shape[:-2] + (1,)
        matrix = rand_boost(shape, std_eta=self.std_eta, device=fourmomenta.device)
        matrix = matrix.expand(*fourmomenta.shape[:-1], 4, 4)

        lframes = LFrames(
            is_global=True,
            matrices=matrix,
            device=fourmomenta.device,
            dtype=fourmomenta.dtype,
        )
        return (lframes, {}) if return_tracker else lframes
