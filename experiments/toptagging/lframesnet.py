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


class LFramesNet(nn.Module):
    def __init__(self, approach, layers, hidden_channels):
        super().__init__()

        self.approach = approach
        if approach == "identity":
            # interpretation: non-equivariant
            self.net = IdentityLFrames()
        elif approach == "random_global":
            # interpretation: data augmentation
            self.net = RandomGlobalLFrames()
        elif approach == "random_local":
            # interpretation: data augmentation + showing off
            self.net = RandomLFrames()
        elif approach == "3nn":
            # interpretation: equivariant
            self.net = ThreeNNLFrames()
        elif approach == "learned_gramschmidt":
            # interpretation: equivariant
            raise NotImplementedError  # not working yet

            hidden_channels = [hidden_channels] * layers
            in_reps = TensorReps("1x0n+1x1n")
            self.net = WrappedLearnedLFrames(
                in_reps=in_reps,
                hidden_channels=hidden_channels,
                radial_module=None,  # TBD
            )
        else:
            raise ValueError(f"approach={self.approach} not implemented")

    def forward(self, x, edge_index, batch):
        if self.approach in ["identity", "random_global", "random_local", "3nn"]:
            pos = x[..., 1:]
            lframes = self.net(pos, idx=None, batch=batch)
        elif self.approach == "learned_gramschmidt":
            raise NotImplementedError

            pos = x[..., 1:]
            lframes = self.net(
                x=x,
                pos=pos,
                edge_index=edge_index,
                edge_attr=None,
                batch=batch,
            )

        return lframes
