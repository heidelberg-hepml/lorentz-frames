import numpy as np
import torch
from torch import nn

from tensorframes.reps import Irreps, TensorReps
from tensorframes.nn.gcn_conv import GCNConv
from tensorframes.lframes.lframes import LFrames


def mean_pointcloud(x, batch):
    batchsize = max(batch) + 1
    logits = torch.zeros(batchsize, device=x.device, dtype=x.dtype)
    logits.index_add_(0, batch, x[:, 0])  # sum
    logits = logits / torch.bincount(batch)  # mean
    return logits


class GCNConvWrapper(nn.Module):
    """
    GCNConv for top-tagging

    Note: Fully-connected convolutional networks are nonsense
    """

    def __init__(
        self,
        lframesnet,
        mean_aggregation,
    ):
        super().__init__()
        self.mean_aggregation = mean_aggregation
        self.lframesnet = lframesnet

        # for proper models we will use hydra more for instantiating
        in_reps = TensorReps("1x0n+1x1n")
        out_reps = TensorReps("1x0n")
        self.net = GCNConv(in_reps, out_reps)

    def forward(self, batch):
        # construct lframes
        lframes = self.lframesnet(batch.x, batch.edge_index, batch.batch)

        # network
        outputs = self.net(edge_index=batch.edge_index, x=batch.x, lframes=lframes)

        # aggregation
        if self.mean_aggregation:
            logits = mean_pointcloud(outputs, batch.batch)
        else:
            logits = outputs[batch.is_global]
        return logits
