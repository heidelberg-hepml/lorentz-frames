import torch
from tensorframes.lframes.lframes import LFrames
from tensorframes.utils.transforms import transform, rand_lorentz, rand_phirotation


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

    def __init__(self, std_eta=1.0):
        super().__init__(is_global=True)
        self.std_eta = std_eta

    def forward(self, fourmomenta, return_tracker=False):
        # general random transformation
        matrix = rand_lorentz([1], std_eta=self.std_eta, device=fourmomenta.device)
        matrix = matrix.repeat(fourmomenta.shape[0], 1, 1)

        lframes = LFrames(
            is_global=True,
            matrices=matrix,
            device=fourmomenta.device,
            dtype=fourmomenta.dtype,
        )
        return (lframes, {}) if return_tracker else lframes


class RandomPhiLFrames(LFramesPredictor):
    """Randomly generates a phi-rotated local frame for the whole batch,
    corresponding to data augmentation with rotations around the z axis."""

    def __init__(self):
        super().__init__(is_global=True)

    def forward(self, fourmomenta, return_tracker=False):
        # random rotation around z axis
        matrix = rand_phirotation([1], device=fourmomenta.device)
        matrix = matrix.repeat(fourmomenta.shape[0], 1, 1)

        lframes = LFrames(
            is_global=True,
            matrices=matrix,
            device=fourmomenta.device,
            dtype=fourmomenta.dtype,
        )
        return (lframes, {}) if return_tracker else lframes
