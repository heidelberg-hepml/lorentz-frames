import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

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


class TaggerWrapper(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        lframesnet,
        add_tagging_features_lframesnet=False,
    ):
        """
        Args:
            in_channels: string representation for the model input
            out_channels: string representation for the model output
            lframesnet: lframesclass
            add_taggin_features_lframesnet: bool whether to include the tagging features in the lframesnet
        """
        super().__init__()
        self.add_tagging_features_lframesnet = add_tagging_features_lframesnet
        # this is the input and output for the net
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lframesnet = lframesnet
        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))

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

        if self.add_tagging_features_lframesnet:
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
        local_tagging_features_nospurions = get_tagging_features(
            fourmomenta_local_nospurions, batch_nospurions
        )

        features_local_nospurions = torch.cat(
            [scalars_nospurions, local_tagging_features_nospurions], dim=-1
        )

        return (
            features_local_nospurions,
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
        self.net = net(features_dims=self.in_channels, num_classes=self.out_channels)

    def forward(self, embedding):
        (
            features_local,
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
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(in_channels=self.in_channels, out_channels=self.out_channels)

    def forward(self, embedding):
        (
            features_local,
            lframes,
            edge_index,
            batch,
            tracker,
        ) = super().forward(embedding)

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
            features_local,
            lframes,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)

        mask = get_xformers_attention_mask(
            batch, materialize=features_local.device == torch.device("cpu")
        )

        # network
        outputs = self.net(inputs=features_local, lframes=lframes, attention_mask=mask)

        # aggregation
        score = self.extract_score(outputs, batch)
        return score, tracker
