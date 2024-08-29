import torch
from torch import nn

from tensorframes.lframes.classical_lframes import (
    IdentityLFrames,
    RandomGlobalLFrames,
    RandomLFrames,
    ThreeNNLFrames,
)
from tensorframes.lframes.learning_lframes import WrappedLearnedLFrames
from tensorframes.reps import TensorReps
from tensorframes.nn.embedding.radial import (
    BesselEmbedding,
    GaussianEmbedding,
    TrivialRadialEmbedding,
)


class LFramesNet(nn.Module):
    def __init__(self, approach, layers, hidden_channels):
        super().__init__()

        self.approach = approach
        if approach == "identity":  # non-equivariant
            self.net = IdentityLFrames()
        elif approach == "random_global":  # data augmentation
            self.net = RandomGlobalLFrames()
        elif approach == "random_local":  # data augmentation + showing off
            self.net = RandomLFrames()
        elif approach == "3nn":  # interpretation: equivariant
            self.net = ThreeNNLFrames()
        elif approach == "learned_gramschmidt":  # interpretation: equivariant
            hidden_channels = [hidden_channels] * layers
            in_reps = TensorReps("1x0n+1x1n")
            radial_module = TrivialRadialEmbedding()  # more options here
            self.net = WrappedLearnedLFrames(
                in_reps=in_reps,
                hidden_channels=hidden_channels,
                radial_module=radial_module,
            )
        else:
            raise ValueError(f"approach={self.approach} not implemented")

    def forward(self, x, edge_index, batch):
        pos = x[..., 1:]
        if self.approach in ["identity", "random_global", "random_local", "3nn"]:
            lframes = self.net(pos, idx=None, batch=batch)
        elif self.approach == "learned_gramschmidt":
            _, lframes = self.net(
                x=x,
                pos=pos,
                edge_index=edge_index,
                edge_attr=None,
                batch=batch,
            )

        return lframes
