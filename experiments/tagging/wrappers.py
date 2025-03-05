import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from functools import partial
from torch_geometric.utils import to_dense_batch

from tensorframes.reps.tensorreps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.utils.hep import EPPP_to_PtPhiEtaM2
from tensorframes.utils.utils import get_xformers_attention_mask
from experiments.tagging.embedding import get_tagging_features


class TaggerWrapper(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        lframesnet,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lframesnet = lframesnet
        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))

    def forward(self, embedding):
        # extract embedding
        fourmomenta = embedding["fourmomenta"]
        scalars = embedding["scalars"]
        edge_index = embedding["edge_index"]
        batch = embedding["batch"]
        ptr = embedding["ptr"]

        # construct lframes
        fourmomenta = fourmomenta.reshape(fourmomenta.shape[0], -1)
        lframes, tracker = self.lframesnet(
            fourmomenta, scalars, ptr=ptr, return_tracker=True
        )

        # transform features into local frames
        fourmomenta_local = self.trafo_fourmomenta(fourmomenta, lframes)
        fourmomenta_local = fourmomenta_local.reshape(
            fourmomenta_local.shape[0],
            -1,
            4,
        )

        return (
            fourmomenta_local,
            scalars,
            lframes,
            edge_index,
            batch,
            tracker,
        )


class AggregatedTaggerWrapper(TaggerWrapper):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aggregator = MeanAggregation()

    def extract_score(self, features, batch):
        score = self.aggregator(features, index=batch)
        return score


class BaselineTransformerWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(in_channels=self.in_channels, num_classes=self.out_channels)
        assert (
            self.lframesnet.is_global
        ), "Non-equivariant model can only handle global lframes"

    def forward(self, embedding):
        fourmomenta_local, scalars, _, _, batch, tracker = super().forward(embedding)
        jetmomenta_local = EPPP_to_PtPhiEtaM2(fourmomenta_local)

        jetmomenta_local = jetmomenta_local.reshape(jetmomenta_local.shape[0], -1)
        features_local = torch.cat([jetmomenta_local, scalars], dim=-1)

        mask = get_xformers_attention_mask(
            batch,
            materialize=features_local.device == torch.device("cpu"),
            dtype=scalars.dtype,
        )

        # network
        outputs = self.net(
            inputs=features_local,
            attention_mask=mask,
        )

        # aggregation
        score = self.extract_score(outputs, batch)
        return score, tracker


class BaselineGraphNetWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(in_channels=self.in_channels, num_classes=self.out_channels)
        assert (
            self.lframesnet.is_global
        ), "Non-equivariant model can only handle global lframes"

    def forward(self, embedding):
        (
            fourmomenta_local,
            scalars,
            _,
            edge_index,
            batch,
            tracker,
        ) = super().forward(embedding)
        jetmomenta_local = EPPP_to_PtPhiEtaM2(fourmomenta_local)

        jetmomenta_local = jetmomenta_local.reshape(jetmomenta_local.shape[0], -1)
        features_local = torch.cat([jetmomenta_local, scalars], dim=-1)

        # network
        outputs = self.net(x=features_local, edge_index=edge_index)

        # aggregation
        score = self.extract_score(outputs, batch)
        return score, tracker


class BaselineParticleNetWrapper(TaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert (
            self.lframesnet.is_global
        ), "Non-equivariant model can only handle global lframes"
        # 7 input features are computed from fourmomenta_local
        # scalars are ignored in this model (for now, thats a design choice)
        num_inputs = 7
        self.net = net(features_dims=num_inputs, num_classes=self.out_channels)

    def forward(self, embedding):
        fourmomenta_local, _, _, _, batch, tracker = super().forward(embedding)
        fourmomenta_local = fourmomenta_local[..., 0, :]
        features_local = get_tagging_features(fourmomenta_local, batch)

        # ParticleNet uses L2 norm in (phi, eta) for kNN
        phieta_local = features_local[..., [4, 5]]

        phieta_local, mask = to_dense_batch(phieta_local, batch)
        features_local, _ = to_dense_batch(features_local, batch)
        phieta_local = phieta_local.transpose(1, 2)
        features_local = features_local.transpose(1, 2)
        mask = mask.unsqueeze(1)

        # network
        score = self.net(
            points=phieta_local,
            features=features_local,
            mask=mask,
        )
        return score, tracker


class BaselineParTWrapper(TaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert (
            self.lframesnet.is_global
        ), "Non-equivariant model can only handle global lframes"
        self.net = net(input_dim=self.in_channels, num_classes=self.out_channels)

    def forward(self, embedding):
        fourmomenta_local, scalars, _, _, batch, tracker = super().forward(embedding)
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
        return score, tracker


class GraphNetWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(in_channels=self.in_channels, out_channels=self.out_channels)

    def forward(self, embedding):
        (
            fourmomenta_local,
            scalars,
            lframes,
            edge_index,
            batch,
            tracker,
        ) = super().forward(embedding)
        jetmomenta_local = EPPP_to_PtPhiEtaM2(fourmomenta_local)

        jetmomenta_local = jetmomenta_local.reshape(jetmomenta_local.shape[0], -1)
        features_local = torch.cat([jetmomenta_local, scalars], dim=-1)

        # network
        outputs = self.net(
            inputs=features_local, lframes=lframes, edge_index=edge_index
        )

        # aggregation
        score = self.extract_score(outputs, batch)
        return score, tracker


class TransformerWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(in_channels=self.in_channels, out_channels=self.out_channels)

    def forward(self, embedding):
        (
            fourmomenta_local,
            scalars,
            lframes,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        jetmomenta_local = EPPP_to_PtPhiEtaM2(fourmomenta_local)

        jetmomenta_local = jetmomenta_local.reshape(jetmomenta_local.shape[0], -1)
        features_local = torch.cat([jetmomenta_local, scalars], dim=-1)

        mask = get_xformers_attention_mask(
            batch, materialize=features_local.device == torch.device("cpu")
        )

        # network
        outputs = self.net(inputs=features_local, lframes=lframes, attention_mask=mask)

        # aggregation
        score = self.extract_score(outputs, batch)
        return score, tracker
