import torch
from torch import nn

from experiments.logger import LOGGER
from torchvision.ops import MLP
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn.aggr import MeanAggregation


class TaggerWrapper(nn.Module):
    def __init__(
        self,
        mean_aggregation,
    ):
        super().__init__()
        self.aggregator = MeanAggregation() if mean_aggregation else None

    def extract_score(self, outputs, batch, is_global):
        if self.aggregator is not None:
            score = self.aggregator(outputs, index=batch)[:, 0]
        else:
            score = outputs[is_global]
        return score

    def forward(self, batch):
        raise NotImplementedError


class LorentzFramesTaggerWrapper(TaggerWrapper):
    def __init__(
        self,
        lframesnet,
        mean_aggregation,
    ):
        super().__init__(mean_aggregation)
        self.lframesnet = lframesnet

    def forward(self, batch):
        raise NotImplementedError


class ProtoNetWrapper(LorentzFramesTaggerWrapper):
    def __init__(
        self,
        net,
        lframesnet,
        mean_aggregation,
        radial_module,
        angular_module,
        in_reps,
        post_layer=None,  # layer to use in the score calculation after the last layer
    ):
        lframesnet = lframesnet(radial_module=radial_module, in_reps=in_reps)
        super().__init__(lframesnet, mean_aggregation)
        self.mean_aggregation = mean_aggregation
        self.net = net(
            radial_module=radial_module, angular_module=angular_module, in_reps=in_reps
        )
        self.post_layer = post_layer
        network_output_dim = self.net.output_dim
        self.in_reps = in_reps
        if self.post_layer is not None:
            assert (
                mean_aggregation == True
            ), "post_layer only works for mean aggregation"
            self.total_post_layers = post_layer
            self.total_post_layers.append(1)
            self.post_layer = MLP(
                in_channels=network_output_dim,
                hidden_channels=self.total_post_layers,
                dropout=0.1,
            )
            LOGGER.debug(f"Using post_layer: {self.post_layer}")

        if not self.mean_aggregation:
            assert (
                network_output_dim == 1
            ), "For global nodes, the output layer should be 1"

    def forward(self, embedding):
        # construct lframes and transform features into them
        batchsize = embedding["fourmomenta"].shape[0]
        x = torch.cat(
            (embedding["scalars"], embedding["fourmomenta"].reshape(batchsize, -1)),
            dim=-1,
        )
        pos = embedding["fourmomenta"][..., 0, 1:]
        edge_index, batch, is_global = [
            embedding[key] for key in ["edge_index", "batch", "is_global"]
        ]
        x_transformed, lframes = self.lframesnet(x, pos, edge_index, batch)
        # network
        outputs = self.net(
            x=x_transformed,
            pos=pos,
            edge_index=edge_index,
            lframes=lframes,
            batch=batch,
        )

        # aggregation
        if self.post_layer is None:
            score = self.extract_score(outputs, batch, is_global)
        else:
            logits = global_mean_pool(outputs, batch)  # size: (batch, output_dim)
            score = self.post_layer(logits)  # size: (batch, 1)
            score = score.flatten()  # size: batch
        return score


class NonEquiNetWrapper(TaggerWrapper):
    def __init__(
        self,
        net,
        mean_aggregation,
        in_reps,
    ):
        super().__init__(mean_aggregation)
        self.net = net(in_reps=in_reps)

    def forward(self, embedding):
        x = torch.cat((embedding["scalars"], embedding["fourmomenta"]), dim=-1)
        pos = embedding["fourmomenta"][..., 1:]
        edge_index, batch, is_global = [
            embedding[key] for key in ["edge_index", "batch", "is_global"]
        ]

        # network
        outputs = self.net(
            x=x,
            pos=pos,
            edge_index=edge_index,
            batch=batch,
        )

        # aggregation
        score = self.extract_score(outputs, batch, is_global)
        return score
