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
            score = outputs[is_global][:, 0]
        return score

    def expand_embedding(self, embedding):
        # construct lframes and transform features into them
        fourmomenta, scalars = embedding["fourmomenta"], embedding["scalars"]
        x = torch.cat(
            (
                scalars,
                fourmomenta.reshape(
                    fourmomenta.shape[0], fourmomenta.shape[1] * fourmomenta.shape[2]
                ),
            ),
            dim=-1,
        )
        pos = fourmomenta[..., 0, :]
        edge_index, batch, is_global = [
            embedding[key] for key in ["edge_index", "batch", "is_global"]
        ]
        return x, pos, edge_index, batch, is_global


class LorentzFramesTaggerWrapper(TaggerWrapper):
    def __init__(
        self,
        lframesnet,
        mean_aggregation,
    ):
        super().__init__(mean_aggregation)
        self.lframesnet = lframesnet


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
        x, pos, edge_index, batch, is_global = self.expand_embedding(embedding)
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
        x, pos, edge_index, batch, is_global = self.expand_embedding(embedding)

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


##########################################################
############### alternative implementation ###############
##########################################################

from functools import partial
from torch_geometric.utils import to_dense_batch
from xformers.ops.fmha import BlockDiagonalMask

from tensorframes.reps.tensorreps import TensorReps
from tensorframes.nnhep.embedding import EPPP_to_PtPhiEtaM2
from tensorframes.lframes.equi_lframes import RestLFrames


def attention_mask(batch, materialize=False):
    """
    Construct attention mask that makes sure that objects only attend to each other
    within the same batch element, and not across batch elements

    Parameters
    ----------
    batch: torch.tensor
        batch object in the torch_geometric.data naming convention
        contains batch index for each event in a sparse tensor
    materialize: bool
        Decides whether a xformers or ('materialized') torch.tensor mask should be returned
        The xformers mask allows to use the optimized xformers attention kernel, but only runs on gpu

    Returns
    -------
    mask: xformers.ops.fmha.attn_bias.BlockDiagonalMask or torch.tensor
        attention mask, to be used in xformers.ops.memory_efficient_attention
        or torch.nn.functional.scaled_dot_product_attention
    """
    bincounts = torch.bincount(batch).tolist()
    mask = BlockDiagonalMask.from_seqlens(bincounts)
    if materialize:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(batch), len(batch))).to(batch.device)
    return mask


class TaggerWrapper2(nn.Module):
    def __init__(
        self,
        in_reps,
        lframesnet,
    ):
        super().__init__()

        self.in_reps = TensorReps(in_reps)
        self.lframesnet = lframesnet
        self.trafo_fourmomenta = TensorReps("1x1n").get_transform_class()

    def forward(self, embedding):
        # extract embedding
        fourmomenta = embedding["fourmomenta"]
        scalars = embedding["scalars"]
        edge_index = embedding["edge_index"]
        batch = embedding["batch"]
        is_global = embedding["is_global"]

        # construct lframes
        fourmomenta = fourmomenta.reshape(fourmomenta.shape[0], -1)
        if self.lframesnet.is_global or isinstance(self.lframesnet, RestLFrames):
            lframes = self.lframesnet(fourmomenta)
        else:
            lframes = self.lframesnet(fourmomenta, scalars, edge_index, batch)

        # transform features into local frames
        fourmomenta_local = self.trafo_fourmomenta(fourmomenta, lframes)
        fourmomenta_local = fourmomenta_local.reshape(
            fourmomenta_local.shape[0],
            -1,
            4,
        )

        return fourmomenta_local, scalars, lframes, edge_index, batch, is_global


class AggregatedTaggerWrapper(TaggerWrapper2):
    def __init__(
        self,
        mean_aggregation,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aggregator = MeanAggregation() if mean_aggregation else None

    def extract_score(self, features, batch, is_global):
        if self.aggregator is not None:
            score = self.aggregator(features, index=batch)[:, 0]
        else:
            score = features[is_global][:, 0]
        return score


class BaselineTransformerWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(in_channels=self.in_reps.dim)
        assert (
            self.lframesnet.is_global
        ), "Non-equivariant model can only handle global lframes"

    def forward(self, embedding):
        fourmomenta_local, scalars, _, _, batch, is_global = super().forward(embedding)
        jetmomenta_local = EPPP_to_PtPhiEtaM2(fourmomenta_local)

        jetmomenta_local = jetmomenta_local.reshape(jetmomenta_local.shape[0], -1)
        features_local = torch.cat([jetmomenta_local, scalars], dim=-1)

        mask = attention_mask(
            batch, materialize=features_local.device == torch.device("cpu")
        )

        # network
        outputs = self.net(
            inputs=features_local,
            attention_mask=mask,
        )

        # aggregation
        score = self.extract_score(outputs, batch, is_global)
        return score


class BaselineGraphNetWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(in_channels=self.in_reps.dim)
        assert (
            self.lframesnet.is_global
        ), "Non-equivariant model can only handle global lframes"

    def forward(self, embedding):
        fourmomenta_local, scalars, _, edge_index, batch, is_global = super().forward(
            embedding
        )
        jetmomenta_local = EPPP_to_PtPhiEtaM2(fourmomenta_local)

        jetmomenta_local = jetmomenta_local.reshape(jetmomenta_local.shape[0], -1)
        features_local = torch.cat([jetmomenta_local, scalars], dim=-1)

        # network
        outputs = self.net(x=features_local, edge_index=edge_index)

        # aggregation
        score = self.extract_score(outputs, batch, is_global)
        return score


class BaselineParticleNetWrapper(TaggerWrapper2):
    def __init__(
        self,
        net,
        mean_aggregation=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert (
            self.lframesnet.is_global
        ), "Non-equivariant model can only handle global lframes"
        assert mean_aggregation
        self.net = net(features_dims=self.in_reps.dim)

    def forward(self, embedding):
        fourmomenta_local, scalars, _, _, batch, is_global = super().forward(embedding)
        jetmomenta_local = EPPP_to_PtPhiEtaM2(fourmomenta_local)

        fourmomenta_local = fourmomenta_local.reshape(fourmomenta_local.shape[0], -1)
        jetmomenta_local = jetmomenta_local.reshape(jetmomenta_local.shape[0], -1)
        features_local = torch.cat([jetmomenta_local, scalars], dim=-1)

        fourmomenta_local, mask = to_dense_batch(fourmomenta_local, batch)
        features_local, _ = to_dense_batch(features_local, batch)
        fourmomenta_local = fourmomenta_local.transpose(1, 2)
        features_local = features_local.transpose(1, 2)
        mask = mask.unsqueeze(1)

        # network
        score = self.net(
            points=fourmomenta_local,
            features=features_local,
            mask=mask,
        )
        return score[:, 0]


class BaselineParTWrapper(TaggerWrapper2):
    def __init__(
        self,
        net,
        mean_aggregation=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert (
            self.lframesnet.is_global
        ), "Non-equivariant model can only handle global lframes"
        assert mean_aggregation
        self.net = net(input_dim=self.in_reps.dim)

    def forward(self, embedding):
        fourmomenta_local, scalars, _, _, batch, is_global = super().forward(embedding)
        jetmomenta_local = EPPP_to_PtPhiEtaM2(fourmomenta_local)

        fourmomenta_local = fourmomenta_local.reshape(fourmomenta_local.shape[0], -1)
        jetmomenta_local = jetmomenta_local.reshape(jetmomenta_local.shape[0], -1)
        features_local = torch.cat([jetmomenta_local, scalars], dim=-1)

        fourmomenta_local, mask = to_dense_batch(fourmomenta_local, batch)
        features_local, _ = to_dense_batch(features_local, batch)
        fourmomenta_local = fourmomenta_local.transpose(1, 2)
        features_local = features_local.transpose(1, 2)
        mask = mask.unsqueeze(1)

        # network
        score = self.net(
            v=fourmomenta_local,
            x=features_local,
            mask=mask,
        )
        return score[:, 0]


class GraphNetWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(in_channels=self.in_reps.dim)

        # lframesnet might not be instantiated yet
        if isinstance(self.lframesnet, partial):
            # learnable lframesnet takes only scalar inputs
            num_scalars = sum(
                rep.mul for rep in self.in_reps if str(rep.rep) in ["0n", "0p"]
            )
            self.lframesnet = self.lframesnet(in_nodes=num_scalars)

    def forward(self, embedding):
        (
            fourmomenta_local,
            scalars,
            lframes,
            edge_index,
            batch,
            is_global,
        ) = super().forward(embedding)
        jetmomenta_local = EPPP_to_PtPhiEtaM2(fourmomenta_local)

        jetmomenta_local = jetmomenta_local.reshape(jetmomenta_local.shape[0], -1)
        features_local = torch.cat([jetmomenta_local, scalars], dim=-1)

        # network
        outputs = self.net(x=features_local, lframes=lframes, edge_index=edge_index)

        # aggregation
        score = self.extract_score(outputs, batch, is_global)
        return score


class TransformerWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(in_channels=self.in_reps.dim)

        # lframesnet might not be instantiated yet
        if isinstance(self.lframesnet, partial):
            # learnable lframesnet takes only scalar inputs
            num_scalars = sum(
                rep.mul for rep in self.in_reps if str(rep.rep) in ["0n", "0p"]
            )
            self.lframesnet = self.lframesnet(in_nodes=num_scalars)

    def forward(self, embedding):
        (
            fourmomenta_local,
            scalars,
            lframes,
            edge_index,
            batch,
            is_global,
        ) = super().forward(embedding)
        jetmomenta_local = EPPP_to_PtPhiEtaM2(fourmomenta_local)

        jetmomenta_local = jetmomenta_local.reshape(jetmomenta_local.shape[0], -1)
        features_local = torch.cat([jetmomenta_local, scalars], dim=-1)

        mask = attention_mask(
            batch, materialize=features_local.device == torch.device("cpu")
        )

        # network
        outputs = self.net(inputs=features_local, lframes=lframes, attention_mask=mask)

        # aggregation
        score = self.extract_score(outputs, batch, is_global)
        return score
