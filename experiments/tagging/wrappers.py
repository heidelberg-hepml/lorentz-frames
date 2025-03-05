import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from functools import partial
from torch_geometric.utils import to_dense_batch
from xformers.ops.fmha import BlockDiagonalMask

from tensorframes.lframes.lframes import LFrames
from experiments.tagging.embedding import get_ptr_from_batch
from tensorframes.reps.tensorreps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from experiments.tagging.embedding import get_tagging_features, get_edge_index_from_ptr
from experiments.logger import LOGGER


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
        in_channels,
        out_channels,
        lframesnet,
        add_tagging_features_lframesnet=False,
        spurion_strategy=None,
    ):
        """
        Args:
            in_channels: string representation for the model input
            out_channels: string representation for the model output
            lframesnet: lframesclass
            add_taggin_features_lframesnet: bool whether to include the tagging features in the lframesnet
            spurion_strategy: {None, "particle_append", "basis_triplet", particle_add"} which spurion_strategy methode to use in the lframesnet, refer to paper
                None: apply no symmetry breaking through vector features
                particle_append: include spurions as elements in the data
                basis_triplet: use the spurions instead of equivariantly predicted vectors in the lframes
                particle_add: add the vectors ot the predicted vectors during prediction of the equivectors
        """
        super().__init__()
        self.add_tagging_features_lframesnet = add_tagging_features_lframesnet
        self.spurion_strategy = spurion_strategy

        # this is the input and output for the net
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert (
            self.out_channels == 1
        ), "out_channels must only contain scalars, but got out_channels={out_channels}"

        if isinstance(lframesnet, partial):
            # lframesnet with learnable elements need the in_nodes (number of scalars in input) for the networks
            self.lframesnet = lframesnet(spurion_strategy=spurion_strategy)

        else:
            self.lframesnet = lframesnet

        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))

    def forward(self, embedding):
        # extract embedding
        fourmomenta_withspurions = embedding["fourmomenta"]
        scalars_withspurions = embedding["scalars"]
        global_tagging_features_withspurions = embedding["global_tagging_features"]
        edge_index_withspurions = embedding["edge_index"]
        batch_withspurions = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        minimal_spurions = embedding["minimal_spurions"]

        assert (
            torch.where(is_spurion)[0]
            == torch.where((fourmomenta_withspurions == 0.0).any(dim=-1))[0]
        ).all()

        # remove spurions from the data again and recompute attributes
        fourmomenta_nospurions = fourmomenta_withspurions[~is_spurion]
        scalars_nospurions = scalars_withspurions[~is_spurion]
        global_tagging_features_nospurions = global_tagging_features_withspurions[
            ~is_spurion
        ]

        possible_spurions = (fourmomenta_nospurions == 0).any(dim=-1)
        if possible_spurions.any():
            LOGGER.info(f"{torch.where(possible_spurions)=}")

        batch_nospurions = batch_withspurions[~is_spurion]
        ptr_nospurions = get_ptr_from_batch(batch_nospurions)
        edge_index_nospurions = get_edge_index_from_ptr(ptr_nospurions)

        if self.lframesnet.is_global:
            lframes_nospurions, tracker = self.lframesnet(
                fourmomenta_nospurions, return_tracker=True
            )
        elif self.spurion_strategy == "particle_append":
            if self.add_tagging_features_lframesnet:
                scalar_features = torch.cat(
                    [scalars_withspurions, global_tagging_features_withspurions], dim=-1
                )
            else:
                scalar_features = scalars_withspurions
            lframes, tracker = self.lframesnet(
                fourmomenta=fourmomenta_withspurions,
                scalars=scalar_features,
                edge_index=edge_index_withspurions,
                batch=batch_withspurions,
                spurions=minimal_spurions,
                return_tracker=True,
            )

            lframes_nospurions = LFrames(
                lframes.matrices[~is_spurion],
                is_global=lframes.is_global,
                det=lframes.det[~is_spurion],
                inv=lframes.inv[~is_spurion],
                is_identity=lframes.is_identity,
            )
        else:
            if self.add_tagging_features_lframesnet:
                scalar_features_nospurions = torch.cat(
                    [scalars_nospurions, global_tagging_features_nospurions], dim=-1
                )
            else:
                scalar_features_nospurions = scalars_nospurions
            lframes_nospurions, tracker = self.lframesnet(
                fourmomenta=fourmomenta_nospurions,
                scalars=scalar_features_nospurions,
                edge_index=edge_index_nospurions,
                batch=batch_nospurions,
                spurions=minimal_spurions,
                return_tracker=True,
            )

        # transform features into local frames
        fourmomenta_local_nospurions = self.trafo_fourmomenta(
            fourmomenta_nospurions, lframes_nospurions
        )
        local_tagging_features = get_tagging_features(
            fourmomenta_local_nospurions, batch_nospurions
        )

        features_local = torch.cat([scalars_nospurions, local_tagging_features], dim=-1)

        # note : this should be removed later, but it seems not to harm performance much
        if not torch.isfinite(features_local).all():
            mask = torch.isfinite(features_local).all(dim=-1)

            a = fourmomenta_nospurions.reshape(-1, 1, 4)
            mat = lframes_nospurions.matrices.reshape(-1, 4, 4)
            output_manual = torch.einsum("aAB,aDB->aDA", mat, a)

            output_manual_element = torch.einsum(
                "AB,DB->DA", mat[~mask][0], a[~mask][0]
            )

            output_trafo = self.trafo_fourmomenta(
                fourmomenta_nospurions[~mask], LFrames(mat[~mask])
            )

            LOGGER.warning(
                f"{fourmomenta_local_nospurions.shape=}, {fourmomenta_nospurions.shape=}, {lframes_nospurions.matrices.shape=}"
            )
            LOGGER.warning(
                f"{scalar_features_nospurions=}, {features_local=}, {features_local[~mask]=}, {torch.where(~mask)=}, {fourmomenta_local_nospurions[~mask]=}"
            )
            LOGGER.warning(
                f"{fourmomenta_nospurions[~mask]=}, {lframes_nospurions.matrices[~mask]=}, {mask.shape=}"
            )
            LOGGER.warning(
                f"{fourmomenta_local_nospurions[0]=}, {fourmomenta_nospurions[0]=}, {lframes_nospurions.matrices[0]=}"
            )
            LOGGER.warning(
                f"{output_manual.shape=}, {output_manual[~mask]=}, {output_trafo.shape=}, {output_trafo=}, {output_manual_element=}"
            )
            assert False

        return (
            features_local,
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

        mask = attention_mask(
            batch, materialize=features_local.device == torch.device("cpu")
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
        # 7 input features are computed from fourmomenta_local
        # scalars are ignored in this model (for now, thats a design choice)
        num_inputs = 7
        self.net = net(features_dims=num_inputs, num_classes=self.out_channels)

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
        mask = attention_mask(
            batch, materialize=features_local.device == torch.device("cpu")
        )

        # network
        outputs = self.net(inputs=features_local, lframes=lframes, attention_mask=mask)

        # aggregation
        score = self.extract_score(outputs, batch)
        return score, tracker
