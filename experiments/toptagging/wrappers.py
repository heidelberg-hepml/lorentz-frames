import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from functools import partial
from torch_geometric.utils import to_dense_batch
from xformers.ops.fmha import BlockDiagonalMask

from tensorframes.reps.tensorreps import TensorReps
from tensorframes.nn.embedding import EPPP_to_PtPhiEtaM2
from tensorframes.lframes.equi_lframes import RestLFrames
from tensorframes.lframes.equi_lframes import LearnedLFrames


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


class TaggerWrapper(nn.Module):
    def __init__(
        self,
        in_reps,
        lframesnet,
    ):
        super().__init__()

        self.in_reps = TensorReps(in_reps)

        if isinstance(lframesnet, partial):
            # lframesnet with learnable elements need the in_nodes (number of scalars in input) for the networks
            if issubclass(lframesnet.func, LearnedLFrames):
                self.lframesnet = lframesnet(in_nodes=self.in_reps.mul_scalars)
            else:
                self.lframesnet = lframesnet
        else:
            self.lframesnet = lframesnet

        self.trafo_fourmomenta = TensorReps("1x1n").get_transform_class()

    def forward(self, embedding):
        # extract embedding
        fourmomenta = embedding["fourmomenta"]
        scalars = embedding["scalars"]
        edge_index = embedding["edge_index"]
        batch = embedding["batch"]
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


class AggregatedTaggerWrapper(TaggerWrapper):
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


class BaselineParticleNetWrapper(TaggerWrapper):
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


class BaselineParTWrapper(TaggerWrapper):
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
        fourmomenta_local, scalars, _, _, batch, _ = super().forward(embedding)
        jetmomenta_local = EPPP_to_PtPhiEtaM2(fourmomenta_local)

        jetmomenta_local = jetmomenta_local.reshape(jetmomenta_local.shape[0], -1)
        features_local = torch.cat([jetmomenta_local, scalars], dim=-1)

        features_local, mask = to_dense_batch(features_local, batch)
        features_local = features_local.transpose(1, 2)
        mask = mask.unsqueeze(1)

        # network
        score = self.net(
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
