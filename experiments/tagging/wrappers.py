import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.utils import scatter

from torch_geometric.utils import to_dense_batch

from tensorframes.lframes.lframes import LFrames
from tensorframes.utils.utils import (
    get_ptr_from_batch,
    get_edge_index_from_ptr,
    get_xformers_attention_mask,
)
from tensorframes.reps.tensorreps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from experiments.tagging.embedding import get_tagging_features
from tensorframes.utils.lorentz import lorentz_squarednorm


class TaggerWrapper(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lframesnet,
        use_float64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lframesnet = lframesnet
        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))
        self.use_float64 = (use_float64,)

    def forward(self, embedding):
        # extract embedding
        fourmomenta_withspurions = embedding["fourmomenta"]
        scalars_withspurions = embedding["scalars"]
        global_tagging_features_withspurions = embedding["global_tagging_features"]
        batch_withspurions = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        ptr_withspurions = embedding["ptr"]

        # remove spurions from the data again and recompute attributes
        fourmomenta_nospurions = fourmomenta_withspurions[~is_spurion]
        scalars_nospurions = scalars_withspurions[~is_spurion]

        batch_nospurions = batch_withspurions[~is_spurion]
        ptr_nospurions = get_ptr_from_batch(batch_nospurions)
        edge_index_nospurions = get_edge_index_from_ptr(ptr_nospurions)

        scalars_withspurions = torch.cat(
            [scalars_withspurions, global_tagging_features_withspurions], dim=-1
        )
        lframes_spurions, tracker = self.lframesnet(
            fourmomenta_withspurions,
            scalars_withspurions,
            ptr=ptr_withspurions,
            return_tracker=True,
        )
        lframes_nospurions = LFrames(
            lframes_spurions.matrices[~is_spurion],
            is_global=lframes_spurions.is_global,
            det=lframes_spurions.det[~is_spurion],
            inv=lframes_spurions.inv[~is_spurion],
            is_identity=lframes_spurions.is_identity,
            device=lframes_spurions.device,
            dtype=lframes_spurions.dtype,
            shape=lframes_spurions.matrices[~is_spurion].shape,
        )

        # transform features into local frames
        fourmomenta_local_nospurions = self.trafo_fourmomenta(
            fourmomenta_nospurions, lframes_nospurions
        )
        jet_nospurions = scatter(
            fourmomenta_nospurions, index=batch_nospurions, dim=0, reduce="sum"
        ).index_select(0, batch_nospurions)
        jet_local_nospurions = self.trafo_fourmomenta(
            jet_nospurions, lframes_nospurions
        )
        local_tagging_features_nospurions = get_tagging_features(
            fourmomenta_local_nospurions,
            jet_local_nospurions,
            use_float64=self.use_float64,
        )

        features_local_nospurions = torch.cat(
            [scalars_nospurions, local_tagging_features_nospurions], dim=-1
        )

        return (
            features_local_nospurions,
            fourmomenta_local_nospurions,
            lframes_nospurions,
            edge_index_nospurions,
            batch_nospurions,
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
        (
            features_local,
            _,
            _,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)

        mask = get_xformers_attention_mask(
            batch,
            materialize=features_local.device == torch.device("cpu"),
            dtype=features_local.dtype,
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
            features_local,
            _,
            _,
            edge_index,
            batch,
            tracker,
        ) = super().forward(embedding)

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
        self.net = net(input_dims=self.in_channels, num_classes=self.out_channels)

    def forward(self, embedding):
        (
            features_local,
            _,
            _,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
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
        (
            features_local,
            _,
            _,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)

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
        include_edges,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.include_edges = include_edges
        self.net = net(in_channels=self.in_channels, out_channels=self.out_channels)
        if self.include_edges:
            self.register_buffer("edge_inited", torch.tensor(False))
            self.register_buffer("edge_mean", torch.tensor(0.0))
            self.register_buffer("edge_std", torch.tensor(1.0))

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            lframes,
            edge_index,
            batch,
            tracker,
        ) = super().forward(embedding)

        if self.include_edges:
            edge_attr = self.get_edge_attr(fourmomenta_local, edge_index)
        else:
            edge_attr = None
        # network
        outputs = self.net(
            inputs=features_local,
            lframes=lframes,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # aggregation
        score = self.extract_score(outputs, batch)
        return score, tracker

    def get_edge_attr(self, fourmomenta, edge_index):
        mij2 = lorentz_squarednorm(
            fourmomenta[edge_index[0]] + fourmomenta[edge_index[1]]
        )
        edge_attr = mij2.clamp(min=1e-10).log()
        if not self.edge_inited:
            self.edge_mean = edge_attr.mean().detach()
            self.edge_std = edge_attr.std().clamp(min=1e-5).detach()
            self.edge_inited = torch.tensor(True, device=edge_attr.device)
        edge_attr = (edge_attr - self.edge_mean) / self.edge_std
        return edge_attr.unsqueeze(-1)


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
            features_local,
            _,
            lframes,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)

        mask = get_xformers_attention_mask(
            batch, materialize=features_local.device == torch.device("cpu")
        )

        # add artificial batch dimension
        features_local = features_local.unsqueeze(0)
        lframes = lframes.reshape(1, *lframes.shape)

        # network
        outputs = self.net(inputs=features_local, lframes=lframes, attention_mask=mask)
        outputs = outputs[0, ...]

        # aggregation
        score = self.extract_score(outputs, batch)
        return score, tracker


class ParticleNetWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(input_dims=self.in_channels, num_classes=self.out_channels)

    def forward(self, embedding):
        (
            features_local,
            _,
            lframes_no_spurions,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        # ParticleNet uses L2 norm in (phi, eta) for kNN
        phieta_local = features_local[..., [4, 5]]

        phieta_local, mask = to_dense_batch(phieta_local, batch)
        features_local, _ = to_dense_batch(features_local, batch)
        phieta_local = phieta_local.transpose(1, 2)
        features_local = features_local.transpose(1, 2)
        lframes_no_spurions = LFrames(
            to_dense_batch(lframes_no_spurions.matrices, batch)[0].view(-1, 4, 4)
        )
        mask = mask.unsqueeze(1)

        # network
        score = self.net(
            points=phieta_local,
            features=features_local,
            lframes=lframes_no_spurions,
            mask=mask,
        )
        return score, tracker
