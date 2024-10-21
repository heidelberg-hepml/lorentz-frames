import torch
from torch import nn

from tensorframes.lframes.classical_lframes import (
    IdentityLFrames,
    RandomGlobalLFrames,
    ThreeNNLFrames,
)
from tensorframes.lframes.learning_lframes import WrappedLearnedLFrames
from tensorframes.reps import TensorReps


class LFramesNet(nn.Module):
    def __init__(self, approach, layers, hidden_channels, radial_module, in_reps):
        super().__init__()
        self.in_reps = in_reps

        self.approach = approach
        if approach == "identity":  # non-equivariant
            self.net = IdentityLFrames()
        elif approach == "random_global":  # data augmentation
            raise NotImplementedError
            self.net = RandomGlobalLFrames()
        elif approach == "3nn":  # interpretation: equivariant
            raise NotImplementedError
            self.net = ThreeNNLFrames()
        elif approach == "learned_gramschmidt":  # interpretation: equivariant
            raise NotImplementedError
            assert radial_module is not None
            hidden_channels = [hidden_channels] * layers
            in_reps = TensorReps(in_reps)
            self.net = WrappedLearnedLFrames(
                in_reps=in_reps,
                hidden_channels=hidden_channels,
                radial_module=radial_module,
            )
        else:
            raise ValueError(f"approach={self.approach} not implemented")

    def forward(self, x, pos, edge_index, batch):
        if self.approach in ["identity", "random_global", "3nn"]:
            lframes = self.net(pos, idx=None, batch=batch)
            trafo = TensorReps(self.in_reps).get_transform_class()
            x_transformed = trafo(x, lframes)

        elif self.approach == "learned_gramschmidt":
            x_transformed, lframes = self.net(
                x=x,
                pos=pos,
                edge_index=edge_index,
                edge_attr=None,
                batch=batch,
            )

        return x_transformed, lframes
