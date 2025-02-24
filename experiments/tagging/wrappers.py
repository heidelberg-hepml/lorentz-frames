import torch
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation

from functools import partial
from torch_geometric.utils import to_dense_batch
from xformers.ops.fmha import BlockDiagonalMask
from torch_geometric.utils import dense_to_sparse

from tensorframes.lframes.lframes import LFrames
from experiments.tagging.embedding import get_ptr_from_batch
from tensorframes.reps.tensorreps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.utils.hep import EPPP_to_PtPhiEtaM2
from tensorframes.lframes.equi_lframes import LearnedLFrames
from experiments.tagging.embedding import get_tagging_features
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
        in_reps,
        out_reps,
        in_nodes,
        lframesnet,
        add_tagging_features_lframesnet,
        symmetry_breaking=[],
    ):
        super().__init__()
        self.add_tagging_features_lframesnet = add_tagging_features_lframesnet
        self.symmetry_breaking = symmetry_breaking

        # this is the input and output for the net
        self.in_reps = TensorReps(in_reps)
        self.out_reps = TensorReps(out_reps)

        assert (
            self.out_reps.mul_without_scalars == 0
        ), "out_reps must only contain scalars, but got out_reps={out_reps}"

        if isinstance(lframesnet, partial):
            # lframesnet with learnable elements need the in_nodes (number of scalars in input) for the networks
            self.lframesnet = lframesnet(
                in_nodes=in_nodes, symmetry_breaking=symmetry_breaking
            )
        else:
            self.lframesnet = lframesnet

        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))

    def forward(self, embedding):
        # extract embedding
        fourmomenta = embedding["fourmomenta"]
        scalars = embedding["scalars"]
        global_tagging_features = embedding["global_tagging_features"]
        edge_index = embedding["edge_index"]
        batch = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        minimal_spurions = embedding["minimal_spurions"]

        # remove spurions from the data again and recompute attributes
        if "vectors" not in self.symmetry_breaking:
            fourmomenta = fourmomenta[~is_spurion]
            scalars = scalars[~is_spurion]
            global_tagging_features = global_tagging_features[~is_spurion]

            batch = batch[~is_spurion]
            ptr = get_ptr_from_batch(batch)
            diffs = torch.diff(ptr)
            edge_index = torch.cat(
                [
                    dense_to_sparse(torch.ones(d, d, device=diffs.device))[0]
                    + diffs[:i].sum()
                    for i, d in enumerate(diffs)
                ],
                dim=-1,
            )

        if self.lframesnet.is_global:
            lframes, tracker = self.lframesnet(fourmomenta, return_tracker=True)
        else:
            if self.add_tagging_features_lframesnet:
                scalar_features = global_tagging_features
            else:
                scalar_features = torch.empty(
                    (fourmomenta.shape[0], 0), device=fourmomenta.device
                )
            lframes, tracker = self.lframesnet(
                fourmomenta,
                scalar_features,
                edge_index,
                spurions=minimal_spurions,
                return_tracker=True,
            )

        # if spurions have not already been removed, we have to do it for the model
        if "vectors" in self.symmetry_breaking:
            fourmomenta = fourmomenta[~is_spurion]
            lframes = LFrames(lframes.matrices[~is_spurion])
            scalars = scalars[~is_spurion]
            batch = batch[~is_spurion]
            ptr = get_ptr_from_batch(batch)
            diffs = torch.diff(ptr)
            edge_index = torch.cat(
                [
                    dense_to_sparse(torch.ones(d, d, device=diffs.device))[0]
                    + diffs[:i].sum()
                    for i, d in enumerate(diffs)
                ],
                dim=-1,
            )

        # transform features into local frames
        fourmomenta_local = self.trafo_fourmomenta(fourmomenta.clone(), lframes)
        local_tagging_features = get_tagging_features(fourmomenta_local, batch)

        features_local = torch.cat([scalars, local_tagging_features], dim=-1)

        # note : this should be removed later, but it seems not to harm performance much
        if not torch.isfinite(features_local).all():
            mask = torch.isfinite(features_local).all(dim=-1)
            LOGGER.warning(
                f"{features_local=}, {features_local[~mask]=}, {torch.where(~mask)=} {fourmomenta_local[~mask]=}"
            )
            assert False

        return (
            features_local,
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
        self.net = net(in_channels=self.in_reps.dim, num_classes=self.out_reps.dim)
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
        self.net = net(in_channels=self.in_reps.dim, num_classes=self.out_reps.dim)
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
        self.net = net(features_dims=num_inputs, num_classes=self.out_reps.dim)

    def forward(self, embedding):
        (
            features_local,
            _,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        features_local = features_local.unsqueeze(1)

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
        self.net = net(input_dim=self.in_reps.dim, num_classes=self.out_reps.dim)

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
        self.net = net(in_reps=self.in_reps, out_reps=self.out_reps)

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

        # note : this should be removed later, but it seems not to harm performance much
        if not torch.isfinite(outputs).all():
            mask = torch.isfinite(outputs).all(dim=-1)
            LOGGER.warning(
                f"{features_local[~mask]=}, {outputs[~mask]=}, {torch.where(~mask)=}"
            )
            assert False
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
        self.net = net(in_reps=self.in_reps, out_reps=self.out_reps)

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
