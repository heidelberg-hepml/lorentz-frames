"""ParticleNet-ParT GraphGPS hybrid: EdgeConv local + ParT-biased attention global.

The rich-edge member of the GraphGPS family for this repo. Each GPS layer fuses a
local and a global branch following the GraphGPS recipe (paper App. D, Eq. 9-11:
per-branch dropout -> residual -> norm, fuse by SUM, then an FFN with its own
residual + norm; FFN inner width = 2*dim):

  * local branch  -- a ParticleNet EdgeConv on a DYNAMIC kNN graph, rebuilt from
    the current hidden features each layer (DGCNN-style). The first layer is seeded
    from the input geometry: (eta, phi) for 'deltaR' or the four-momenta for
    'minkowski'. EdgeConv owns an internal shortcut residual, so -- exactly as
    GraphGPS does for its CustomGatedGCN local model -- the GPS layer adds only the
    external normalization on this branch, not a second residual.

  * global branch -- ParT attention: plain torch ``nn.MultiheadAttention`` plus the
    ParT pairwise interaction bias U_ij (a ``PairEmbed`` of the four-momenta). This is
    exactly GraphGPS's 'BiasedTransformer' slot (``attn_bias``). For non-trivial frames
    the q/k/v are transported between local frames by a parameter-free ``LLoCaAttention``
    reusing the layer's own MHA weights; for identity frames it is just the plain MHA.

Compared with ParticleNetParTGraphTrans (a sequential EdgeConv stage -> ParT stage),
this interleaves the two per layer. Like the rest of the GraphGPS family it uses
BatchNorm and a mean-pool readout (no class token), so no jet frame is needed.

LLoCa is added *purely additively* (matching the library's tensorial message-passing):
the EdgeConv transports neighbours (``change_local_frame``, typed by ``edge_reps``) and
the attention transports q/k/v (``LLoCaAttention``, typed by ``attn_reps``) only when the
wrapper (``...ParticleNetParTGraphGPSWrapper``) passes non-trivial per-particle frames.
For identity/global frames every transport is skipped and the backbone is bit-identical to
the plain, non-equivariant hybrid (the transport adds no parameters and no init randomness).

Input convention (channels-first, matching ParT/ParticleNet/PlainGraphGPS):
    points:   (N, 2, P)   kNN coordinates (eta, phi), seeds layer 0 for 'deltaR'
    features: (N, C, P)   scalar per-particle features
    v:        (N, 4, P)   four-momenta as (px, py, pz, energy); pairwise bias + 'minkowski'
    mask:     (N, 1, P)   1 for real particles, 0 for padding
"""

import torch
import torch.nn as nn
from lloca.backbone.attention import LLoCaAttention
from lloca.reps.tensorreps import TensorReps

from experiments.baselines.particlenettransformer import (
    EdgeConvBlock,
    PairEmbed,
    lloca_transport_attention,
)
from experiments.baselines.plaingraphgps import MaskedNorm

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU}


class ParTGPSLayer(nn.Module):
    """One GraphGPS layer with an EdgeConv local branch and ParT-biased attention.

    The EdgeConv (like GraphGPS's GatedGCN) carries its own internal residual, so
    only the external norm is applied to the local branch; the attention branch
    gets the standard external dropout -> residual -> norm (Eq. 10).
    """

    def __init__(self, dim, num_heads, edge_k, edge_mlp_layers=2, ffn_ratio=2,
                 dropout=0.0, attn_dropout=0.0, act="relu", norm="batch",
                 for_inference=False, edge_reps=None):
        super().__init__()
        Act = _ACT[act]
        self.edge_k = edge_k
        # local: ParticleNet EdgeConv (in==out -> identity shortcut, i.e. internal residual).
        # edge_reps types the hidden features for the LLoCa neighbour transport (additive;
        # no-op for identity frames). None -> non-tensorial original EdgeConv.
        self.local = EdgeConvBlock(
            k=edge_k, in_feat=dim, out_feats=[dim] * edge_mlp_layers, cpu_mode=for_inference,
            in_reps=edge_reps,
        )
        # global: plain MHA; ParT pairwise bias is the additive attn_mask. For learned frames
        # the q/k/v are transported by a shared LLoCaAttention reusing these MHA weights.
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.num_heads = num_heads

        self.norm_local = MaskedNorm(dim, norm)
        self.norm_attn = MaskedNorm(dim, norm)
        self.norm_ffn = MaskedNorm(dim, norm)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_ratio * dim), Act(),
            nn.Dropout(dropout), nn.Linear(ffn_ratio * dim, dim),
        )

    def forward(self, h, coord, knn_metric, key_padding_mask, node_mask, mask_knn,
                attn_bias=None, frames_flat=None, lloca_attn=None):
        # h: (B, P, C); coord: (B, Cc, P) or None (None -> dynamic graph on h);
        # node_mask: (B, P, 1); mask_knn: (B, P); attn_bias: (B, num_heads, P, P);
        # frames_flat: (B*P, 4, 4) for the EdgeConv transport; lloca_attn: shared LLoCaAttention
        # (None on the identity/global path -> plain attention, bit-identical to the original).
        mask_bool = ~key_padding_mask
        h_cf = (h * node_mask).transpose(1, 2)                       # (B, C, P)

        # ---- local branch (Eq. 9): EdgeConv owns its residual -> external norm only ----
        kcoord = coord if coord is not None else h_cf
        kmetric = knn_metric if coord is not None else "deltaR"      # deeper layers: feature-space L2
        h_local = self.local(kcoord, h_cf, knn_metric=kmetric, mask=mask_knn, frames=frames_flat)
        h_local = self.norm_local(h_local.transpose(1, 2), mask_bool)          # (B, P, C)

        # ---- global branch (Eq. 10): ParT-biased attention (LLoCa transport for learned frames) ----
        if lloca_attn is None:
            am = attn_bias
            if am is not None and am.dim() == 4:
                am = am.reshape(-1, am.size(-2), am.size(-1))        # (B*num_heads, P, P)
            a = self.attn(h, h, h, attn_mask=am, key_padding_mask=key_padding_mask,
                          need_weights=False)[0]
        else:
            a = lloca_transport_attention(
                h, self.attn, lloca_attn, key_padding_mask=key_padding_mask,
                attn_mask=attn_bias, dropout_p=self.attn.dropout if self.training else 0.0,
            )
        h_attn = self.norm_attn(self.drop_attn(a) + h, mask_bool)

        # ---- fuse by SUM, then FFN (Eq. 11) ----
        h = h_local + h_attn
        h = self.norm_ffn(self.drop_ffn(self.ffn(h)) + h, mask_bool)
        return h * node_mask


class ParticleNetParTGraphGPS(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # dynamic kNN graph for the EdgeConv local branch
                 knn_k=16,
                 knn_metric="deltaR",       # how layer-0 graph is seeded ('deltaR'/'minkowski')
                 edge_mlp_layers=2,
                 use_fts_bn=True,
                 # ParT pairwise attention bias (global branch)
                 bias=True,                 # toggle the ParT pairwise additive attention bias
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 pair_embed_dims=[64, 64, 64],
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 # GPS layers
                 dim=128,
                 num_layers=10,
                 num_heads=8,
                 # LLoCa tensorial message-passing (additive; a no-op for identity/global frames).
                 # attn_reps types the per-head q/k/v transport (attn_reps.dim * num_heads == dim);
                 # edge_reps types the EdgeConv hidden features (edge_reps.dim == dim). Shared by
                 # every layer. None -> non-tensorial (the original GraphGPS hybrid).
                 attn_reps="8x0n+2x1n",
                 edge_reps=None,
                 ffn_ratio=2,
                 dropout=0.0,
                 attn_dropout=0.0,
                 act="relu",
                 norm="batch",
                 # readout
                 head_layers=2,
                 for_inference=False,
                 use_amp=False,
                 **kwargs):
        super().__init__(**kwargs)
        if knn_metric not in ("deltaR", "minkowski"):
            raise ValueError(f"knn_metric must be 'deltaR' or 'minkowski', got '{knn_metric}'")
        self.knn_k = knn_k
        self.knn_metric = knn_metric
        self.num_heads = num_heads
        self.for_inference = for_inference
        self.use_amp = use_amp
        Act = _ACT[act]

        # shared, parameter-free LLoCaAttention for the attention transport (learned frames only)
        self.lloca_attn = None
        if attn_reps is not None:
            attn_reps_t = TensorReps(attn_reps)
            assert attn_reps_t.dim * num_heads == dim, f"{attn_reps_t.dim}*{num_heads} != dim {dim}"
            self.lloca_attn = LLoCaAttention(attn_reps_t, num_heads)

        self.bn_fts = nn.BatchNorm1d(input_dim) if use_fts_bn else None
        self.node_encoder = nn.Linear(input_dim, dim)

        # ParT pairwise bias: one shared PairEmbed, last dim = num_heads (one bias per head)
        self.pair_embed = None
        if bias and pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0:
            self.pair_embed = PairEmbed(
                pair_input_dim, pair_extra_dim, pair_embed_dims + [num_heads],
                remove_self_pair=remove_self_pair,
                use_pre_activation_pair=use_pre_activation_pair, for_onnx=for_inference,
            )

        self.layers = nn.ModuleList([
            ParTGPSLayer(dim, num_heads, knn_k, edge_mlp_layers, ffn_ratio,
                         dropout, attn_dropout, act, norm, for_inference, edge_reps=edge_reps)
            for _ in range(num_layers)
        ])

        # SAN-style readout: mean pool -> dim-halving MLP -> logits
        head, d = [], dim
        for _ in range(head_layers):
            head += [nn.Linear(d, d // 2), Act()]
            d //= 2
        head += [nn.Linear(d, num_classes)]
        self.head = nn.Sequential(*head)

    def forward(self, points, features, v=None, mask=None, frames=None):
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)
        else:
            mask = mask.bool()
        points = points * mask
        features = features * mask
        mask_p = mask.squeeze(1)                     # (B, P)
        P = features.size(-1)

        # LLoCa transport is additive: engaged only for non-trivial frames. The readout is a
        # mean-pool over the (invariant) local features, so no jet frame is needed.
        do_transport = frames is not None and not frames.is_global
        frames_flat = frames.reshape(-1, 4, 4) if do_transport else None
        if do_transport:
            if self.lloca_attn is None:
                raise ValueError(
                    "learned frames require an attention transport, but attn_reps is None "
                    "(the attention branch has no tensorial reps to transport q/k/v). Set "
                    "model.net.attn_reps, or use identity frames."
                )
            self.lloca_attn.prepare_frames(frames)
        block_lloca = self.lloca_attn if do_transport else None

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # ParT pairwise interaction bias U_ij, computed once and shared across layers.
            # PairEmbed pads a CLS slot at index 0; we mean-pool (no CLS) so we drop it. Keep
            # it (B, num_heads, P, P); the layer reshapes to (B*num_heads, P, P) for plain MHA.
            attn_bias = None
            if self.pair_embed is not None and v is not None:
                attn_bias = self.pair_embed(v)[:, :, 1:, 1:]        # (B, num_heads, P, P)

            fts = self.bn_fts(features) * mask if self.bn_fts is not None else features
            h = self.node_encoder(fts.transpose(1, 2))              # (B, P, dim)

            node_mask = mask_p.unsqueeze(-1).to(h.dtype)            # (B, P, 1)
            key_padding_mask = ~mask_p                              # (B, P), True = ignore
            for i, layer in enumerate(self.layers):
                if i == 0:
                    coord = v if self.knn_metric == "minkowski" and v is not None else points
                    metric = self.knn_metric
                else:
                    coord, metric = None, "deltaR"                  # dynamic feature-space graph
                h = layer(h, coord, metric, key_padding_mask, node_mask, mask_p, attn_bias,
                          frames_flat, block_lloca)

            # masked mean pooling over real particles
            pooled = (h * node_mask).sum(dim=1) / node_mask.sum(dim=1).clamp(min=1.0)
            output = self.head(pooled)
            if self.for_inference:
                # single-logit (binary, BCE-style) heads must use sigmoid: softmax over a
                # 1-wide dim is identically 1.0 (constant score -> silent AUC 0.5)
                output = (torch.sigmoid(output) if output.shape[1] == 1
                          else torch.softmax(output, dim=1))
            return output
