import torch
from tensorframes.lframes.lframes import LFrames
from tensorframes.utils.transforms import transform, rand_transform, rand_phirotation


class LFramesPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class IdentityLFrames(LFramesPredictor):
    """Identity local frames, corresponding to non-equivariant networks"""

    def __init__(self):
        super().__init__()

    def forward(self, fourmomenta):
        return LFrames.global_trafo(fourmomenta.device, n_batch=fourmomenta.shape[0])


class RandomLFrames(LFramesPredictor):
    """Randomly generates a local frame for the whole batch,
    corresponding to data augmentation."""

    def __init__(self, std_eta=1.0):
        super().__init__()
        self.std_eta = std_eta

    def forward(self, fourmomenta):
        # general random transformation
        matrix = rand_transform([1], std_eta=self.std_eta, device=fourmomenta.device)

        return LFrames.global_trafo(
            device=fourmomenta.device, trafo=matrix, n_batch=fourmomenta.shape[0]
        )


class RandomPhiLFrames(LFramesPredictor):
    """Randomly generates a phi-rotated local frame for the whole batch,
    corresponding to data augmentation with rotations around the z axis."""

    def __init__(self):
        super().__init__()

    def forward(self, fourmomenta):
        # random rotation around z axis
        matrix = rand_phirotation([1], device=fourmomenta.device)

        return LFrames.global_trafo(
            device=fourmomenta.device, trafo=matrix, n_batch=fourmomenta.shape[0]
        )
