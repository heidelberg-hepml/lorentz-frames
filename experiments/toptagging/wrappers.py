import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from tensorframes.reps import TensorReps
from tensorframes.nn.gcn_conv import GCNConv
from tensorframes.lframes.lframes import LFrames
from experiments.toptagging.protonet import ProtoNet


class LorentzFramesTaggerWrapper(nn.Module):
    def __init__(
        self,
        lframesnet,
        mean_aggregation,
    ):
        super().__init__()
        self.aggregator = MeanAggregation() if mean_aggregation else None
        self.lframesnet = lframesnet

    def extract_score(self, outputs, batch):
        if self.aggregator is not None:
            score = self.aggregator(outputs, index=batch.batch)[:, 0]
        else:
            score = outputs[batch.is_global]
        return score

    def forward(self, batch):
        raise NotImplementedError


class GCNConvWrapper(LorentzFramesTaggerWrapper):
    """
    GCNConv for top-tagging

    Note: Fully-connected convolutional networks are nonsense
    """

    def __init__(
        self,
        lframesnet,
        mean_aggregation,
    ):
        super().__init__(lframesnet, mean_aggregation)

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
        score = self.extract_score(outputs, batch)
        return score


class ProtoNetWrapper(LorentzFramesTaggerWrapper):
    def __init__(
        self,
        net,
        lframesnet,
        mean_aggregation,
        radial_module
    ):
        lframesnet = lframesnet(radial_module=radial_module)
        super().__init__(lframesnet, mean_aggregation)
        self.net = net

    def forward(self, batch):
        # construct lframes
        lframes = self.lframesnet(batch.x, batch.edge_index, batch.batch)

        # network
        pos = batch.x[:, 1:]
        outputs = self.net(
            x=batch.x,
            pos=pos,
            edge_index=batch.edge_index,
            lframes=lframes,
            batch=batch.batch,
        )

        # aggregation
        score = self.extract_score(outputs, batch)
        return score


class ProtoNet2Wrapper(LorentzFramesTaggerWrapper):
    def __init__(
        self,
        net,
        lframesnet,
        mean_aggregation,
        radial_module,
        angular_module
    ):
        lframesnet = lframesnet(radial_module=radial_module)
        super().__init__(lframesnet, mean_aggregation)
        self.net = net(radial_module=radial_module, angular_module=angular_module)


    def forward(self, batch):
        # construct lframes
        lframes = self.lframesnet(batch.x, batch.edge_index, batch.batch)

        # network
        pos = batch.x[:, 1:]
        outputs = self.net(
            x=batch.x,
            pos=pos,
            edge_index=batch.edge_index,
            lframes=lframes,
            batch=batch.batch,
        )

        # aggregation
        score = self.extract_score(outputs, batch)
        return score

class ReferenceNetWrapper(nn.Module):
    def __init__(
        self,
        net,
        mean_aggregation,
    ):
        super().__init__()
        self.net = net
        self.mean_aggregation = mean_aggregation

    def forward(self, batch):

        # network
        pos = batch.x[:, 1:]
        outputs = self.net(
            x=batch.x,
            pos=pos,
            edge_index=batch.edge_index,
            batch=batch.batch,
        )

        # aggregation
        score = self.extract_score(outputs, batch)
        return score

    def extract_score(self, outputs, batch):
        if self.mean_aggregation:
            score = mean_pointcloud(outputs, batch.batch)
        else:
            score = outputs[batch.is_global]
        return score
