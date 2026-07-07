import torch
from lgatr import embed_vector, extract_scalar
from lloca.framesnet.frames import Frames
from lloca.framesnet.nonequi_frames import IdentityFrames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.lorentz import lorentz_eye
from lloca.utils.orthogonalize_4d import orthogonalize_4d
from lloca.utils.utils import (
    get_batch_from_ptr,
    get_edge_attr,
    get_edge_index_from_ptr,
    get_ptr_from_batch,
)
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.utils import scatter, to_dense_batch

from experiments.misc import get_attention_mask
from experiments.tagging.embedding import get_tagging_features


class TaggerWrapper(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        framesnet,
        add_fourmomenta_backbone: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_fourmomenta_backbone = add_fourmomenta_backbone
        self.framesnet = framesnet
        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))
        # subclasses that need a single covariant per-event frame (e.g. for a prepended
        # readout token in a tensorial backbone) set this; forward then fills _jet_frames.
        self.compute_jet_frames = False
        self._jet_frames = None

    def jet_frames(self, fourmomenta, scalars, ptr):
        """A single covariant frame per event: the boost into the jet rest frame.

        orthogonalize_4d makes the first of its three reference vectors the timelike axis,
        so we pass the jet four-momentum (the sum of the particle four-momenta) as that
        first vector -- the frame's time axis is then the jet direction (its rest frame).
        The two remaining axes (the spatial orientation, which a single momentum leaves
        undetermined) are fixed covariantly by the framesnet's per-particle equivariant
        reference vectors, averaged over the event. All inputs are covariant, so the frame
        is covariant and the readout stays Lorentz-invariant. Returns Frames (B, 4, 4), or
        None for a non-learned (identity) framesnet.
        """
        fn = self.framesnet
        if not hasattr(fn, "equivectors"):
            return None  # IdentityFrames etc. -> identity readout frame (handled downstream)
        batch = get_batch_from_ptr(ptr)
        jet_p = scatter(fourmomenta, batch, dim=0, reduce="sum")  # (B, 4): jet four-momentum
        jet_p = fn.mass_regularize(jet_p)
        # set-level equivectors (e.g. pelican) require num_graphs; the main framesnet path
        # passes it, so mirror that here (B events for these no-spurion particles).
        vecs = fn.equivectors(
            fn.mass_regularize(fourmomenta),
            scalars=scalars,
            ptr=ptr,
            num_graphs=ptr.numel() - 1,
        )
        vecs = scatter(vecs, batch, dim=0, reduce="mean")  # (B, n_vectors, 4) per event
        # time axis = jet momentum; spatial orientation from the (covariant) equivectors
        vecs = torch.cat([jet_p.unsqueeze(1), vecs[:, :2]], dim=1)  # (B, 3, 4)
        # jet_frames always uses the 4d orthogonalizer, but reuses the framesnet's
        # ortho_kwargs. The PD-family framesnets (LearnedPD/SO3/Rest/SO2/Z) key the coplanar
        # regulator as `eps_reg` -- the name their internal orthogonalize_3d expects -- whereas
        # orthogonalize_4d names it `eps_reg_coplanar`. Translate so any framesnet works here.
        ortho_kwargs = dict(fn.ortho_kwargs)
        if "eps_reg" in ortho_kwargs:
            ortho_kwargs.setdefault("eps_reg_coplanar", ortho_kwargs.pop("eps_reg"))
        trafo = orthogonalize_4d(vecs, **ortho_kwargs)  # (B, 4, 4)
        return Frames(trafo.to(fourmomenta.dtype))

    def init_standardization(self, fourmomenta, ptr, reduce_size=None):
        # framesnet equivectors edge_attr standardization (if applicable)
        if hasattr(self.framesnet, "equivectors") and hasattr(
            self.framesnet.equivectors, "init_standardization"
        ):
            fourmomenta_reduced = (
                fourmomenta[:reduce_size] if reduce_size is not None else fourmomenta
            )
            self.framesnet.equivectors.init_standardization(fourmomenta_reduced, ptr)

    def forward(self, embedding):
        # extract embedding
        fourmomenta_withspurions = embedding["fourmomenta"]
        scalars_withspurions = embedding["scalars"]
        global_tagging_features_withspurions = embedding["tagging_features"]
        batch_withspurions = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        ptr_withspurions = embedding["ptr"]
        num_graphs = embedding["num_graphs"]
        nospurion_idxs = (~is_spurion).nonzero(as_tuple=False).squeeze(-1)

        # remove spurions from the data again and recompute attributes
        fourmomenta_nospurions = fourmomenta_withspurions.index_select(0, nospurion_idxs)
        scalars_nospurions = scalars_withspurions.index_select(0, nospurion_idxs)

        batch_nospurions = batch_withspurions.index_select(0, nospurion_idxs)
        ptr_nospurions = get_ptr_from_batch(batch_nospurions)
        B = ptr_nospurions.numel() - 1

        scalars_withspurions = torch.cat(
            [scalars_withspurions, global_tagging_features_withspurions], dim=-1
        )
        frames_spurions, tracker = self.framesnet(
            fourmomenta_withspurions,
            scalars_withspurions,
            ptr=ptr_withspurions,
            return_tracker=True,
            num_graphs=num_graphs,
        )
        matrices = frames_spurions.matrices.index_select(0, nospurion_idxs)
        frames_nospurions = Frames(
            matrices,
            is_global=frames_spurions.is_global,
            det=frames_spurions.det.index_select(0, nospurion_idxs),
            inv=frames_spurions.inv.index_select(0, nospurion_idxs),
            is_identity=frames_spurions.is_identity,
            device=frames_spurions.device,
            dtype=frames_spurions.dtype,
            shape=matrices.shape,
        )

        # optional single covariant per-event (jet) frame, from the real particles only
        if self.compute_jet_frames:
            self._jet_frames = self.jet_frames(
                fourmomenta_nospurions,
                scalars_withspurions.index_select(0, nospurion_idxs),
                ptr_nospurions,
            )

        # transform features into local frames
        fourmomenta_local_nospurions = self.trafo_fourmomenta(
            fourmomenta_nospurions, frames_nospurions
        )
        jet_nospurions = scatter(
            fourmomenta_nospurions,
            index=batch_nospurions,
            dim=0,
            reduce="sum",
            dim_size=B,
        ).index_select(0, batch_nospurions)
        jet_local_nospurions = self.trafo_fourmomenta(jet_nospurions, frames_nospurions)
        local_tagging_features_nospurions = get_tagging_features(
            fourmomenta_local_nospurions,
            jet_local_nospurions,
            tagging_features="all",
        )

        features_local_nospurions = torch.cat(
            [scalars_nospurions, local_tagging_features_nospurions], dim=-1
        )
        if self.add_fourmomenta_backbone:
            features_local_nospurions = torch.cat(
                [features_local_nospurions, fourmomenta_local_nospurions], dim=-1
            )

        # change dtype (see embedding.py fourmomenta_float64 option)
        features_local_nospurions = features_local_nospurions.to(scalars_nospurions.dtype)
        frames_nospurions.to(scalars_nospurions.dtype)

        return (
            features_local_nospurions,
            fourmomenta_local_nospurions,
            frames_nospurions,
            ptr_nospurions,
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

    def extract_score(self, features, ptr):
        B = ptr.numel() - 1
        score = self.aggregator(features, ptr=ptr, dim_size=B)
        return score


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
            frames,
            ptr,
            batch,
            tracker,
        ) = super().forward(embedding)

        edge_index = get_edge_index_from_ptr(ptr, features_local.shape, remove_self_loops=True)
        if self.include_edges:
            edge_attr = self.get_edge_attr(fourmomenta_local, edge_index).to(features_local.dtype)
        else:
            edge_attr = None
        # network
        outputs = self.net(
            inputs=features_local,
            frames=frames,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # aggregation
        score = self.extract_score(outputs, ptr)
        return score, tracker, frames

    def get_edge_attr(self, fourmomenta, edge_index):
        edge_attr = get_edge_attr(fourmomenta, edge_index)
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
        use_amp=False,
        attention_backend="xformers",
        mean_aggregation=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.attention_backend = attention_backend
        self.mean_aggregation = mean_aggregation
        self.net = net(in_channels=self.in_channels, out_channels=self.out_channels)

        if attention_backend == "flex":
            compile_flex_attention(package_name="lloca")

    def forward(self, embedding):
        # precompute attention mask to avoid cudaStreamSynchronize
        # from .tolist() in get_xformers_attention_mask
        batch_withspurions = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        nospurion_idxs = (~is_spurion).nonzero(as_tuple=False).squeeze(-1)
        batch_nospurions = batch_withspurions.index_select(0, nospurion_idxs)
        ptr_nospurions = get_ptr_from_batch(batch_nospurions)
        ptr, batch = ptr_nospurions, batch_nospurions
        if not self.mean_aggregation:
            batchsize = len(ptr) - 1
            ptr = ptr.clone()
            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr)
        mask_kwarg = get_attention_mask(
            batch,
            dtype=embedding["scalars"].dtype,
            attention_backend=self.attention_backend,
        )

        (
            features_local,
            _,
            frames,
            ptr,
            batch,
            tracker,
        ) = super().forward(embedding)

        # handle global token
        if self.mean_aggregation:
            is_global = None
        else:
            # append global tokens to batch, ptr, features_local and frames
            # and keep a is_global mask for later extraction
            batchsize = len(ptr) - 1
            global_idxs = ptr[:-1] + torch.arange(batchsize, device=batch.device)
            is_global = torch.zeros(
                features_local.shape[0] + batchsize,
                dtype=torch.bool,
                device=ptr.device,
            )
            is_global[global_idxs] = True
            features_local_buffer = features_local.clone()
            features_local = torch.zeros(
                is_global.shape[0],
                *features_local.shape[1:],
                dtype=features_local.dtype,
                device=features_local.device,
            )
            features_local[~is_global] = features_local_buffer
            is_global_channel = torch.zeros(
                features_local.shape[0],
                1,
                dtype=features_local.dtype,
                device=features_local.device,
            )
            is_global_channel[is_global] = 1
            features_local = torch.cat((features_local, is_global_channel), dim=-1)

            # global token frames are identity
            matrices_new = (
                torch.eye(4, device=frames.device, dtype=frames.dtype)
                .unsqueeze(0)
                .expand(is_global.shape[0], -1, -1)
            ).clone()
            matrices_new[~is_global] = frames.matrices
            det_new = torch.ones(
                is_global.shape[0], device=frames.device, dtype=frames.dtype
            ).clone()
            det_new[~is_global] = frames.det
            inv_new = (
                torch.eye(4, device=frames.device, dtype=frames.dtype)
                .unsqueeze(0)
                .expand(is_global.shape[0], -1, -1)
            ).clone()
            inv_new[~is_global] = frames.inv
            frames = Frames(
                matrices_new,
                is_global=frames.is_global,
                det=det_new,
                inv=inv_new,
                is_identity=frames.is_identity,
                device=frames.device,
                dtype=frames.dtype,
                shape=matrices_new.shape,
            )

            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr)

        # add artificial batch dimension
        features_local = features_local.unsqueeze(0)
        frames = frames.reshape(1, *frames.shape)

        # network
        with torch.autocast("cuda", enabled=self.use_amp):
            outputs = self.net(inputs=features_local, frames=frames, **mask_kwarg)

        # aggregation
        outputs = outputs[0, ...]
        if self.mean_aggregation:
            score = self.extract_score(outputs, ptr)
        else:
            score = outputs[is_global]
        return score, tracker, frames


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
            frames,
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
        dense_frames, _ = to_dense_batch(frames.matrices, batch)
        dense_frames[~mask] = (
            torch.eye(4, device=dense_frames.device, dtype=dense_frames.dtype)
            .unsqueeze(0)
            .expand((~mask).sum(), -1, -1)
        )

        frames = Frames(
            dense_frames.view(-1, 4, 4),
            is_global=frames.is_global,
            is_identity=frames.is_identity,
            device=frames.device,
            dtype=frames.dtype,
            shape=frames.matrices.shape,
        )
        mask = mask.unsqueeze(1)

        # network
        score = self.net(
            points=phieta_local,
            features=features_local,
            frames=frames,
            mask=mask,
        )
        return score, tracker, frames


class LGATrWrapper(nn.Module):
    def __init__(
        self,
        net,
        framesnet,
        out_channels,
        mean_aggregation=False,
        use_amp=False,
        attention_backend="xformers",
    ):
        super().__init__()
        self.use_amp = use_amp
        self.attention_backend = attention_backend
        self.net = net(out_mv_channels=out_channels)
        self.aggregator = MeanAggregation() if mean_aggregation else None

        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

        if attention_backend == "flex":
            compile_flex_attention(package_name="lgatr")

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20

        # handle global token
        if self.aggregator is None:
            batchsize = len(ptr) - 1
            global_idxs = ptr[:-1] + torch.arange(batchsize, device=batch.device)
            is_global = torch.zeros(
                fourmomenta.shape[0] + batchsize,
                dtype=torch.bool,
                device=ptr.device,
            )
            is_global[global_idxs] = True
            fourmomenta_buffer = fourmomenta.clone()
            fourmomenta = torch.zeros(
                is_global.shape[0],
                *fourmomenta.shape[1:],
                dtype=fourmomenta.dtype,
                device=fourmomenta.device,
            )
            fourmomenta[~is_global] = fourmomenta_buffer
            scalars_buffer = scalars.clone()
            scalars = torch.zeros(
                fourmomenta.shape[0],
                scalars.shape[1] + 1,
                dtype=scalars.dtype,
                device=scalars.device,
            )
            token_idx = torch.nn.functional.one_hot(torch.arange(1, device=scalars.device))
            token_idx = token_idx.repeat(batchsize, 1)
            scalars[~is_global] = torch.cat(
                (
                    scalars_buffer,
                    torch.zeros(
                        scalars_buffer.shape[0],
                        token_idx.shape[1],
                        dtype=scalars.dtype,
                        device=scalars.device,
                    ),
                ),
                dim=-1,
            )
            scalars[is_global] = torch.cat(
                (
                    torch.zeros(
                        token_idx.shape[0],
                        scalars_buffer.shape[1],
                        dtype=scalars.dtype,
                        device=scalars.device,
                    ),
                    token_idx,
                ),
                dim=-1,
            )
            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr)
        else:
            is_global = None

        fourmomenta = fourmomenta.unsqueeze(0).to(scalars.dtype)
        scalars = scalars.unsqueeze(0)

        mask_kwarg = get_attention_mask(
            batch,
            dtype=scalars.dtype,
            attention_backend=self.attention_backend,
        )

        mv = embed_vector(fourmomenta).unsqueeze(-2)
        s = scalars if scalars.shape[-1] > 0 else None

        with torch.autocast("cuda", enabled=self.use_amp):
            mv_outputs, _ = self.net(mv, s, **mask_kwarg)
        out = extract_scalar(mv_outputs)[0, :, :, 0]

        if self.aggregator is not None:
            B = ptr.numel() - 1
            logits = self.aggregator(out, index=batch, dim_size=B)
        else:
            logits = out[is_global]
        return logits, {}, None


class ParTWrapper(TaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        use_amp=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(input_dim=self.in_channels, num_classes=self.out_channels, use_amp=use_amp)

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # need (px, py, pz, E)

        features_local, mask = to_dense_batch(features_local, batch)
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        features_local = features_local.transpose(1, 2)
        fourmomenta_local = fourmomenta_local.transpose(1, 2)

        frames_matrices, _ = to_dense_batch(frames.matrices, batch)
        det, _ = to_dense_batch(frames.det, batch)
        inv, _ = to_dense_batch(frames.inv, batch)
        frames_matrices[~mask] = lorentz_eye(
            frames_matrices[~mask].shape[:-2],
            device=frames.device,
            dtype=frames.dtype,
        )
        frames = Frames(
            matrices=frames_matrices,
            is_global=frames.is_global,
            det=det,
            inv=inv,
            is_identity=frames.is_identity,
            device=frames.device,
            dtype=frames.dtype,
            shape=frames.matrices.shape,
        )

        mask = mask.unsqueeze(1).float()

        # network
        score = self.net(
            x=features_local,
            frames=frames,
            v=fourmomenta_local,
            mask=mask,
        )
        return score, tracker, frames


class MIParTWrapper(ParTWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.framesnet, IdentityFrames)

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
            _,
            batch,
            tracker,
        ) = super(ParTWrapper, self).forward(embedding)
        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # need (px, py, pz, E)

        features_local, mask = to_dense_batch(features_local, batch)
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        features_local = features_local.transpose(1, 2)
        fourmomenta_local = fourmomenta_local.transpose(1, 2)
        mask = mask.unsqueeze(1).float()

        # network
        score = self.net(
            x=features_local,
            v=fourmomenta_local,
            mask=mask,
        )
        return score, tracker, frames


class LorentzNetWrapper(nn.Module):
    def __init__(
        self,
        net,
        framesnet,
        out_channels,
    ):
        super().__init__()
        self.net = net(n_class=out_channels)

        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20

        edge_index = get_edge_index_from_ptr(ptr, fourmomenta.shape, remove_self_loops=True)
        fourmomenta = fourmomenta.to(scalars.dtype)
        output = self.net(scalars, fourmomenta, edges=edge_index, batch=batch)
        return output, {}, None


class PELICANWrapper(nn.Module):
    def __init__(
        self,
        net,
        framesnet,
        out_channels,
    ):
        super().__init__()
        self.net = net(out_channels=out_channels)

        self.register_buffer("edge_inited", torch.tensor(False))
        self.register_buffer("edge_mean", torch.tensor(0.0))
        self.register_buffer("edge_std", torch.tensor(1.0))

        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]
        num_graphs = embedding["num_graphs"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20

        edge_index = get_edge_index_from_ptr(ptr, fourmomenta.shape, remove_self_loops=False)
        fourmomenta = fourmomenta.to(scalars.dtype)
        edge_attr = self.get_edge_attr(fourmomenta, edge_index).to(scalars.dtype)
        output = self.net(
            in_rank2=edge_attr,
            edge_index=edge_index,
            batch=batch,
            in_rank1=scalars,
            num_graphs=num_graphs,
        )
        return output, {}, None

    def get_edge_attr(self, fourmomenta, edge_index):
        edge_attr = get_edge_attr(fourmomenta, edge_index)
        if not self.edge_inited:
            self.edge_mean = edge_attr.mean().detach()
            self.edge_std = edge_attr.std().clamp(min=1e-5).detach()
            self.edge_inited = torch.tensor(True, device=edge_attr.device)
        edge_attr = (edge_attr - self.edge_mean) / self.edge_std
        return edge_attr.unsqueeze(-1)


class PELICANWrapperOfficial(nn.Module):
    def __init__(self, net, framesnet, out_channels):
        super().__init__()
        self.net = net(out_channels=out_channels)
        self.framesnet = framesnet
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20
        fourmomenta = fourmomenta.to(scalars.dtype)
        fourmomenta, mask = to_dense_batch(fourmomenta, batch)
        scalars, _ = to_dense_batch(scalars, batch)
        mask = mask.unsqueeze(-1)

        output = self.net(scalars, fourmomenta, mask=mask)
        return output, {}, None


class CGENNWrapper(nn.Module):
    def __init__(self, net, framesnet, out_channels):
        super().__init__()
        self.net = net(n_outputs=out_channels)
        self.framesnet = framesnet
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        # we mimic the CGENN wrapper of
        # https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/blob/master/models/lorentz_cggnn.py

        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20
        fourmomenta = fourmomenta.to(scalars.dtype)
        zeros = torch.zeros(scalars.shape[0], 1, device=scalars.device, dtype=scalars.dtype)
        scalars = torch.cat((scalars, zeros), dim=-1)

        # pad to dense tensors
        fourmomenta, mask = to_dense_batch(fourmomenta, batch)
        scalars, _ = to_dense_batch(scalars, batch)
        batch_size, n_nodes, _ = fourmomenta.shape
        # Build the fully-connected (no self-loop) edges among the REAL nodes of each jet
        # in the DENSE b*n_nodes+i frame that the flattened tensors below actually use.
        # The previous get_edge_index_from_ptr(ptr, ...) call produced edges in the SPARSE
        # ptr[b]+i frame; the two frames only agree when every jet in the batch has the
        # same length, so with variable jet sizes ~1/3 of the edges crossed jet boundaries
        # or pointed at padded slots, silently scrambling the CGENN message passing and
        # leaking information across batch-mates.
        pair = mask[:, :, None] & mask[:, None, :]
        pair &= ~torch.eye(n_nodes, dtype=torch.bool, device=mask.device)[None]
        b_idx, i_idx, j_idx = pair.nonzero(as_tuple=True)
        edge_index = torch.stack([b_idx * n_nodes + i_idx, b_idx * n_nodes + j_idx])
        fourmomenta = fourmomenta.view(batch_size * n_nodes, -1)
        scalars = scalars.view(batch_size * n_nodes, -1)
        mask = mask.view(batch_size * n_nodes, -1)

        x = fourmomenta.unsqueeze(-2)
        i, j = edge_index
        edge_attr_x = torch.cat(
            [
                x[i],
                x[j],
                x[i] - x[j],
            ],
            dim=-2,
        )
        node_attr_x = x
        x = embed_vector(x)
        edge_attr_x = embed_vector(edge_attr_x)
        node_attr_x = embed_vector(node_attr_x)

        h = scalars
        edge_attr_h = None
        node_attr_h = h

        out = self.net(
            h=h,
            x=x,
            edge_attr_x=edge_attr_x,
            node_attr_x=node_attr_x,
            edge_attr_h=edge_attr_h,
            node_attr_h=node_attr_h,
            edges=edge_index,
            n_nodes=n_nodes,
            node_mask=mask,
        )

        return out, {}, None


class LGATrSlimWrapper(nn.Module):
    def __init__(
        self,
        net,
        framesnet,
        out_channels,
        mean_aggregation=False,
        attention_backend="xformers",
        use_amp=False,
    ):
        super().__init__()
        self.use_amp = use_amp
        self.attention_backend = attention_backend
        self.net = net(out_s_channels=out_channels)
        self.aggregator = MeanAggregation() if mean_aggregation else None
        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

        if attention_backend == "flex":
            compile_flex_attention(package_name="lgatr")

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20

        # handle global token
        if self.aggregator is None:
            batchsize = len(ptr) - 1
            global_idxs = ptr[:-1] + torch.arange(batchsize, device=batch.device)
            is_global = torch.zeros(
                fourmomenta.shape[0] + batchsize,
                dtype=torch.bool,
                device=ptr.device,
            )
            is_global[global_idxs] = True
            fourmomenta_buffer = fourmomenta.clone()
            fourmomenta = torch.zeros(
                is_global.shape[0],
                *fourmomenta.shape[1:],
                dtype=fourmomenta.dtype,
                device=fourmomenta.device,
            )
            fourmomenta[~is_global] = fourmomenta_buffer
            scalars_buffer = scalars.clone()
            scalars = torch.zeros(
                fourmomenta.shape[0],
                scalars.shape[1] + 1,
                dtype=scalars.dtype,
                device=scalars.device,
            )
            token_idx = torch.nn.functional.one_hot(torch.arange(1, device=scalars.device))
            token_idx = token_idx.repeat(batchsize, 1)
            scalars[~is_global] = torch.cat(
                (
                    scalars_buffer,
                    torch.zeros(
                        scalars_buffer.shape[0],
                        token_idx.shape[1],
                        dtype=scalars.dtype,
                        device=scalars.device,
                    ),
                ),
                dim=-1,
            )
            scalars[is_global] = torch.cat(
                (
                    torch.zeros(
                        token_idx.shape[0],
                        scalars_buffer.shape[1],
                        dtype=scalars.dtype,
                        device=scalars.device,
                    ),
                    token_idx,
                ),
                dim=-1,
            )
            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr)
        else:
            is_global = None

        fourmomenta = fourmomenta.unsqueeze(0).to(scalars.dtype)
        scalars = scalars.unsqueeze(0)

        mask_kwarg = get_attention_mask(
            batch,
            dtype=fourmomenta.dtype,
            attention_backend=self.attention_backend,
        )

        v = fourmomenta.unsqueeze(-2)
        s = scalars

        with torch.autocast("cuda", enabled=self.use_amp):
            _, out_s = self.net(v, s, **mask_kwarg)
        out = out_s[0, :, :]

        if self.aggregator is not None:
            logits = self.aggregator(out, index=batch)
        else:
            logits = out[is_global]
        return logits, {}, None


class CGENNLGATrGraphTransWrapper(nn.Module):
    def __init__(self, net, framesnet, out_channels):
        super().__init__()
        self.net = net(num_classes=out_channels)
        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)
    def forward(self, embedding):
        fourmomenta = embedding["fourmomenta"]                 # (E, px, py, pz), incl. spurions
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        keep = ~is_spurion                                     # channel-spurions in model: drop the tokens
        fourmomenta = fourmomenta[keep]
        scalars = scalars[keep]
        batch = batch[keep]
        fourmomenta = (fourmomenta / 20).to(scalars.dtype)     # match the equivariant baselines; NO reorder
        px, py, pz = fourmomenta[:, 1], fourmomenta[:, 2], fourmomenta[:, 3]   # (E, px, py, pz)
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)
        fourmomenta, mask = to_dense_batch(fourmomenta, batch)
        scalars, _ = to_dense_batch(scalars, batch)
        points, _ = to_dense_batch(points, batch)
        output = self.net(
            scalars,
            fourmomenta,
            mask,
            points,
        )
        return output, {}, None


class CGENNLGATrGraphGPSWrapper(nn.Module):
    """Wrapper for the equivariant CGENN-L-GATr GraphGPS hybrid.

    Equivariant by construction (CGENN + L-GATr, symmetry broken only by the model's
    own input spurions), so no LLoCa canonicalization: inherits nn.Module with the
    identity framesnet, exactly like CGENNLGATrGraphTransWrapper. Drops the token
    spurions (the model injects its own as mv channels), rescales by 1/20, and keeps
    the time-first (E, px, py, pz) convention (no reorder).
    """

    def __init__(self, net, framesnet, out_channels):
        super().__init__()
        self.net = net(num_classes=out_channels)
        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        fourmomenta = embedding["fourmomenta"]                 # (E, px, py, pz), incl. spurions
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        keep = ~is_spurion                                     # channel-spurions in model: drop the tokens
        fourmomenta = fourmomenta[keep]
        scalars = scalars[keep]
        batch = batch[keep]
        fourmomenta = (fourmomenta / 20).to(scalars.dtype)     # match the equivariant baselines; NO reorder
        px, py, pz = fourmomenta[:, 1], fourmomenta[:, 2], fourmomenta[:, 3]   # (E, px, py, pz)
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)
        fourmomenta, mask = to_dense_batch(fourmomenta, batch)
        scalars, _ = to_dense_batch(scalars, batch)
        points, _ = to_dense_batch(points, batch)
        output = self.net(
            scalars,
            fourmomenta,
            mask,
            points,
        )
        return output, {}, None


class ParticleNetParTGraphTransWrapper(TaggerWrapper):
    """Wrapper for the ParticleNet-ParT graph-transformer hybrid.

    Non-equivariant, made Lorentz-equivariant by LLoCa tensorial message-passing
    (matching the library), exactly like the other GraphTrans/GPS wrappers: the inputs
    are canonicalized and the per-particle frames are passed into the backbone, which
    transports the EdgeConv neighbours and the attention q/k/v between frames. With
    IdentityFrames the transport is a no-op and this reduces to the plain baseline in the
    global frame; any learned framesnet is supported through the shared TaggerWrapper
    machinery.

    The backbone differs from the rest of the repo only in its conventions: it is
    channels-first (N, C, P), expects four-momenta as (px, py, pz, E) rather than
    (E, px, py, pz), and takes a (N, 1, P) mask. It additionally needs (eta, phi)
    points for the EdgeConv kNN, which we read off the local four-momenta.
    """

    def __init__(self, net, *args, use_amp=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.net = net(input_dim=self.in_channels, num_classes=self.out_channels, use_amp=use_amp)
        # the prepended class token rides in the covariant jet frame -> request it
        self.compute_jet_frames = True

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # need (px, py, pz, E)

        # (eta, phi) points for the EdgeConv kNN, read off the local four-momenta
        px, py, pz = (
            fourmomenta_local[..., 0],
            fourmomenta_local[..., 1],
            fourmomenta_local[..., 2],
        )
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)

        features_local, mask = to_dense_batch(features_local, batch)
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        points, _ = to_dense_batch(points, batch)
        features_local = features_local.transpose(1, 2).contiguous()  # (B, C, P)
        fourmomenta_local = fourmomenta_local.transpose(1, 2).contiguous()  # (B, 4, P)
        points = points.transpose(1, 2).contiguous()  # (B, 2, P)

        # densify the per-particle local frames to (B, P, 4, 4); padded particles get the
        # identity frame (they are masked out of the kNN and attention anyway). The
        # tensorial backbone transports neighbours/q-k-v between these frames (LLoCa).
        frames_matrices, _ = to_dense_batch(frames.matrices, batch)
        frames_matrices[~mask] = lorentz_eye(
            frames_matrices[~mask].shape[:-2], device=frames.device, dtype=frames.dtype
        )
        dense_frames = Frames(
            matrices=frames_matrices,
            is_global=frames.is_global,
            is_identity=frames.is_identity,
        )
        mask = mask.unsqueeze(1).float()  # (B, 1, P)

        # the backbone handles AMP internally via use_amp. cls_frames is the covariant
        # jet frame for the prepended class token (None for IdentityFrames -> identity slot).
        score = self.net(
            points=points,
            features=features_local,
            v=fourmomenta_local,
            frames=dense_frames,
            mask=mask,
            cls_frames=self._jet_frames,
        )
        return score, tracker, frames


class LorentzNetLGATrSlimGraphTransWrapper(nn.Module):
    """Wrapper for the internally-equivariant LorentzNet -> L-GATr-slim hybrid.

    Like CGENNLGATrGraphTransWrapper, the backbone is Lorentz-equivariant by
    construction (LorentzNet GNN + L-GATr-slim, with symmetry broken only by its
    own input-stage spurions), so no LLoCa canonicalization is applied and the
    framesnet must be the identity -- hence we inherit nn.Module directly.

    The backbone differs from the rest of the repo only in conventions: it is
    channels-first (N, C, P), expects four-momenta as (px, py, pz, E) rather than
    (E, px, py, pz), takes a (N, 1, P) mask, and uses (eta, phi) points only when
    knn_metric='deltaR'.
    """

    def __init__(self, net, framesnet, out_channels):
        super().__init__()
        self.net = net(num_classes=out_channels)
        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        fourmomenta = embedding["fourmomenta"]  # (E, px, py, pz), incl. spurions
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        is_spurion = embedding["is_spurion"]

        # the model injects its own input-stage spurions: drop the token spurions
        keep = ~is_spurion
        fourmomenta = fourmomenta[keep]
        scalars = scalars[keep]
        batch = batch[keep]

        # match the scale of the other equivariant baselines
        fourmomenta = (fourmomenta / 20).to(scalars.dtype)

        # (eta, phi) points for the deltaR kNN option (ignored for minkowski)
        px, py, pz = fourmomenta[:, 1], fourmomenta[:, 2], fourmomenta[:, 3]
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)

        # the model expects four-momenta as (px, py, pz, E)
        fourmomenta = fourmomenta[:, [1, 2, 3, 0]]

        # densify and switch to the (N, C, P) channels-first convention
        fourmomenta, mask = to_dense_batch(fourmomenta, batch)  # (B, P, 4), (B, P)
        scalars, _ = to_dense_batch(scalars, batch)  # (B, P, C)
        points, _ = to_dense_batch(points, batch)  # (B, P, 2)

        output = self.net(
            scalars.transpose(1, 2).contiguous(),  # x: (B, C, P)
            fourmomenta.transpose(1, 2).contiguous(),  # v: (B, 4, P)
            mask.unsqueeze(1),  # (B, 1, P)
            points.transpose(1, 2).contiguous(),  # (B, 2, P)
        )
        return output, {}, None


class LorentzNetLGATrSlimGraphGPSWrapper(nn.Module):
    """Wrapper for the equivariant LorentzNet-L-GATr-slim GraphGPS hybrid.

    Equivariant by construction (broken only by the model's own input spurions),
    so no LLoCa canonicalization: inherits nn.Module + identity framesnet, exactly
    like LorentzNetLGATrSlimGraphTransWrapper. Drops the token spurions, rescales
    by 1/20, and reorders four-momenta to the (px, py, pz, E) the model expects.
    """

    def __init__(self, net, framesnet, out_channels):
        super().__init__()
        self.net = net(num_classes=out_channels)
        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        fourmomenta = embedding["fourmomenta"]  # (E, px, py, pz), incl. spurions
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        is_spurion = embedding["is_spurion"]

        # the model injects its own input-stage spurions: drop the token spurions
        keep = ~is_spurion
        fourmomenta = fourmomenta[keep]
        scalars = scalars[keep]
        batch = batch[keep]

        # match the scale of the other equivariant baselines
        fourmomenta = (fourmomenta / 20).to(scalars.dtype)

        # (eta, phi) points for the deltaR kNN option (ignored for minkowski)
        px, py, pz = fourmomenta[:, 1], fourmomenta[:, 2], fourmomenta[:, 3]
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)

        # the model expects four-momenta as (px, py, pz, E)
        fourmomenta = fourmomenta[:, [1, 2, 3, 0]]

        # densify and switch to the (N, C, P) channels-first convention
        fourmomenta, mask = to_dense_batch(fourmomenta, batch)  # (B, P, 4), (B, P)
        scalars, _ = to_dense_batch(scalars, batch)  # (B, P, C)
        points, _ = to_dense_batch(points, batch)  # (B, P, 2)

        output = self.net(
            scalars.transpose(1, 2).contiguous(),  # x: (B, C, P)
            fourmomenta.transpose(1, 2).contiguous(),  # v: (B, 4, P)
            mask.unsqueeze(1),  # (B, 1, P)
            points.transpose(1, 2).contiguous(),  # (B, 2, P)
        )
        return output, {}, None


class PlainGraphTransWrapper(TaggerWrapper):
    """Wrapper for the plain graph-transformer (static MPNN + torch-MHA encoder).

    Non-equivariant, made Lorentz-equivariant by LLoCa tensorial message-passing
    (matching the library), exactly like ParticleNetParTGraphTransWrapper: the inputs
    are canonicalized and the per-particle frames are passed into the backbone, which
    transports the MPNN neighbours and the attention q/k/v between frames. Channels-first
    (N, C, P), four-momenta as (px, py, pz, E), a (N, 1, P) mask, and (eta, phi) points
    for the deltaR kNN. The prepended class token rides in the covariant jet frame.
    """

    def __init__(self, net, *args, use_amp=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.net = net(input_dim=self.in_channels, num_classes=self.out_channels, use_amp=use_amp)
        # the prepended class token rides in the covariant jet frame -> request it
        self.compute_jet_frames = True

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # need (px, py, pz, E)

        px, py, pz = (
            fourmomenta_local[..., 0],
            fourmomenta_local[..., 1],
            fourmomenta_local[..., 2],
        )
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)

        features_local, mask = to_dense_batch(features_local, batch)
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        points, _ = to_dense_batch(points, batch)

        # densify the per-particle local frames to (B, P, 4, 4); padded particles -> identity.
        # The tensorial backbone transports neighbours/q-k-v between these frames (no-op for
        # IdentityFrames -> bit-identical to the plain backbone).
        frames_matrices, _ = to_dense_batch(frames.matrices, batch)
        frames_matrices[~mask] = lorentz_eye(
            frames_matrices[~mask].shape[:-2], device=frames.device, dtype=frames.dtype
        )
        dense_frames = Frames(
            matrices=frames_matrices,
            is_global=frames.is_global,
            is_identity=frames.is_identity,
        )

        score = self.net(
            points=points.transpose(1, 2).contiguous(),  # (B, 2, P)
            features=features_local.transpose(1, 2).contiguous(),  # (B, C, P)
            v=fourmomenta_local.transpose(1, 2).contiguous(),  # (B, 4, P)
            mask=mask.unsqueeze(1).float(),  # (B, 1, P)
            frames=dense_frames,
            cls_frames=self._jet_frames,
        )
        return score, tracker, frames


class PlainGraphGPSWrapper(TaggerWrapper):
    """Wrapper for the plain GraphGPS hybrid (interleaved static-MPNN + torch-MHA).

    Non-equivariant, made Lorentz-equivariant by LLoCa tensorial message-passing
    (matching the library): the inputs are canonicalized and the per-particle frames are
    passed into the backbone, which transports the local-MPNN neighbours and the attention
    q/k/v between frames. Channels-first (N, C, P), four-momenta as (px, py, pz, E), a
    (N, 1, P) mask, and (eta, phi) points for the deltaR kNN. No jet frame is needed -- the
    mean-pool readout over invariant local features is already invariant.
    """

    def __init__(self, net, *args, use_amp=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.net = net(input_dim=self.in_channels, num_classes=self.out_channels, use_amp=use_amp)

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # need (px, py, pz, E)

        px, py, pz = (
            fourmomenta_local[..., 0],
            fourmomenta_local[..., 1],
            fourmomenta_local[..., 2],
        )
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)

        features_local, mask = to_dense_batch(features_local, batch)
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        points, _ = to_dense_batch(points, batch)

        # densify the per-particle local frames to (B, P, 4, 4); padded particles -> identity.
        # The tensorial backbone transports neighbours/q-k-v between these frames (no-op for
        # IdentityFrames -> bit-identical to the plain backbone).
        frames_matrices, _ = to_dense_batch(frames.matrices, batch)
        frames_matrices[~mask] = lorentz_eye(
            frames_matrices[~mask].shape[:-2], device=frames.device, dtype=frames.dtype
        )
        dense_frames = Frames(
            matrices=frames_matrices,
            is_global=frames.is_global,
            is_identity=frames.is_identity,
        )

        score = self.net(
            points=points.transpose(1, 2).contiguous(),  # (B, 2, P)
            features=features_local.transpose(1, 2).contiguous(),  # (B, C, P)
            v=fourmomenta_local.transpose(1, 2).contiguous(),  # (B, 4, P)
            frames=dense_frames,
            mask=mask.unsqueeze(1).float(),  # (B, 1, P)
        )
        return score, tracker, frames


class ParticleNetParTGraphGPSWrapper(TaggerWrapper):
    """Wrapper for the ParticleNet-ParT GraphGPS hybrid (tensorial EdgeConv + ParT attn).

    Lorentz-equivariant by LLoCa tensorial message-passing (matching the library): the
    inputs are canonicalized and the per-particle frames are passed into the backbone,
    which transports neighbours (EdgeConv) and q/k/v (attention) between frames. Like the
    GraphTrans wrapper it is channels-first (N, C, P), four-momenta as (px, py, pz, E), a
    (N, 1, P) mask, and (eta, phi) points seeding the layer-0 deltaR kNN. No jet frame is
    needed -- the mean-pool readout over invariant local features is already invariant.
    """

    def __init__(self, net, *args, use_amp=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.net = net(input_dim=self.in_channels, num_classes=self.out_channels, use_amp=use_amp)

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # need (px, py, pz, E)

        px, py, pz = (
            fourmomenta_local[..., 0],
            fourmomenta_local[..., 1],
            fourmomenta_local[..., 2],
        )
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)

        features_local, mask = to_dense_batch(features_local, batch)
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        points, _ = to_dense_batch(points, batch)

        # densify the per-particle local frames to (B, P, 4, 4); padded particles -> identity
        frames_matrices, _ = to_dense_batch(frames.matrices, batch)
        frames_matrices[~mask] = lorentz_eye(
            frames_matrices[~mask].shape[:-2], device=frames.device, dtype=frames.dtype
        )
        dense_frames = Frames(
            matrices=frames_matrices,
            is_global=frames.is_global,
            is_identity=frames.is_identity,
        )

        score = self.net(
            points=points.transpose(1, 2).contiguous(),  # (B, 2, P)
            features=features_local.transpose(1, 2).contiguous(),  # (B, C, P)
            v=fourmomenta_local.transpose(1, 2).contiguous(),  # (B, 4, P)
            frames=dense_frames,
            mask=mask.unsqueeze(1).float(),  # (B, 1, P)
        )
        return score, tracker, frames


def compile_flex_attention(package_name="lgatr"):
    """Run torch.compile on the flex_attention function.

    However, as of today (Dec 2025, pytorch 2.9.0), torch.compile + flex_attention
    for variable-length sequences only works in a few cases:
    - CPU: Forward pass is supported, but backward pass not (https://github.com/pytorch/pytorch/issues/169224)
      To still let the code run through for tests, we skip torch.compile on CPU.
      This way the code runs through, but is super slow because it materializes the attention matrix.
      Note that we use essentially the same approach for xformers, where we fall back to default torch attention on CPU.
      On the plus side, flex_attention supports arbitrary head_dim if torch.compile is not used.
    - GPU: The docs say that only head dimensions being powers of 2 are supported.
      However, on my system only head_dim=2**n with n>=4 works, i.e. head_dim=16,32,...
      Setting head_dim=2,4,8 gives cryptic errors.
      Moreover, transformers with flex_attention are still significantly slower than
      transformers with xformers attention in our implementation.
    """
    if package_name == "lgatr":
        import lgatr.primitives.attention_backends.flex as flex
    elif package_name == "lloca":
        import lloca.backbone.attention_backends.flex as flex
    else:
        raise ValueError(f"Unknown package {package_name}")

    if torch.cuda.is_available():
        # max-autotune strongly recommended for flex-attention with variable-length sequences,
        # see https://pytorch.org/blog/flexattention-for-inference/
        flex.attention = torch.compile(
            flex.attention,
            dynamic=True,
        )
