import torch
from torch import nn

from tensorframes.lframes.classical_lframes import (
    IdentityLFrames,
    RandomGlobalLFrames,
    NNLFrames,
    COMLFrames,
    PartialCOMLFrames,
    RestLFrames,
    PartialRestLFrames,
)
from tensorframes.lframes.learning_lframes import WrappedLearnedLFrames
from tensorframes.reps import TensorReps


class LFramesNet(nn.Module):
    def __init__(
        self, approach, layers, hidden_channels, radial_module, in_reps, **kwargs
    ):
        super().__init__()
        self.in_reps = in_reps

        self.approach = approach
        if approach == "identity":  # non-equivariant
            self.net = IdentityLFrames()
        elif approach == "random_global":  # data augmentation
            mean_eta = kwargs.get("mean_eta", 0)
            std_eta = kwargs.get("std_eta", 1)
            self.net = RandomGlobalLFrames(mean_eta=mean_eta, std_eta=std_eta)
        elif approach == "nn":  # interpretation: equivariant
            self.net = NNLFrames()
        elif approach == "learned_gramschmidt":  # interpretation: equivariant
            assert radial_module is not None
            hidden_channels = [hidden_channels] * layers
            in_reps = TensorReps(in_reps)
            self.net = WrappedLearnedLFrames(
                in_reps=in_reps,
                hidden_channels=hidden_channels,
                radial_module=radial_module,
                predict_4=False,
            )
        elif approach == "COM":
            self.net = COMLFrames()
        elif approach == "partialCOM":
            self.net = PartialCOMLFrames()
        elif approach == "Rest":
            self.net = RestLFrames()
        elif approach == "partialRest":
            self.net = PartialRestLFrames()
        else:
            raise ValueError(f"approach={self.approach} not implemented")

    def forward(self, x, pos, edge_index, batch):
        if self.approach in [
            "identity",
            "random_global",
            "nn",
            "COM",
            "partialCOM",
            "Rest",
            "partialRest",
        ]:
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
