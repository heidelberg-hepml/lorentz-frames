"""Plain graph-transformer: static plain-MPNN on a kNN graph -> torch-MHA encoder.

The minimal, non-equivariant member of the graph-transformer family used in this
repo. It deliberately uses only the basic GraphGPS components:

  * a STATIC kNN graph (built once per forward, reused by every GNN block), with
    a selectable metric -- 'deltaR' (L2 on (eta, phi)) or 'minkowski'
    (|interval| on the four-momenta); both carry the same robustness tweaks as
    the other hybrids (k capped at P-1, padded nodes excluded);
  * plain message-passing blocks with the default update function
    (message MLP on [h_i, h_j] -> mean aggregation -> update MLP on [h_i, agg],
    with a residual) -- no equivariant structure, no edge-attention gate;
  * a plain transformer encoder of torch ``nn.MultiheadAttention`` blocks
    (NOT L-GATr, no pairwise interaction bias);

with an optional raw-input skip, a linear bridge, and a learnable class token
read out for classification.

Being non-equivariant, it is made Lorentz-equivariant through LLoCa input
canonicalization in PlainGraphTransWrapper (which inherits TaggerWrapper), like
the ParT / ParticleNet baselines and ParticleNetParTGraphTrans.

Input convention (channels-first, matching ParT/ParticleNet):
    points:   (N, 2, P)   kNN coordinates (eta, phi), used when knn_metric='deltaR'
    features: (N, C, P)   scalar per-particle features
    v:        (N, 4, P)   four-momenta as (px, py, pz, energy), used for 'minkowski'
    mask:     (N, 1, P)   1 for real particles, 0 for padding
"""


import torch
import torch.nn as nn
from lloca.backbone.attention import LLoCaAttention
from lloca.backbone.particlenet import change_local_frame
from lloca.framesnet.frames import Frames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.lorentz import lorentz_eye

from experiments.baselines.particlenettransformer import lloca_transport_attention


def knn(x, k, metric='deltaR', mask=None):
    # cap k so topk never exceeds the available points (small/sparse events)
    num_points = x.size(-1)
    k = min(k, max(1, num_points - 1))
    if metric == 'minkowski':
        # x: (N, 4, P) in (px, py, pz, E); rank by |(x_i - x_j)^2|_minkowski
        sig = x.new_tensor([-1.0, -1.0, -1.0, 1.0]).view(1, -1, 1)
        xp = x * sig
        msq = (x * xp).sum(dim=1, keepdim=True)  # (N, 1, P)
        gram = torch.matmul(x.transpose(2, 1), xp)  # (N, P, P)
        d2 = msq.transpose(2, 1) + msq - 2 * gram
        pairwise_distance = -d2.abs()
        eye = torch.eye(num_points, dtype=torch.bool, device=x.device).unsqueeze(0)
        pairwise_distance = pairwise_distance.masked_fill(eye, float('-inf'))
        if mask is not None:
            pairwise_distance = pairwise_distance.masked_fill(~mask.bool().unsqueeze(1), float('-inf'))
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
    else:
        # 'deltaR': squared-L2 on the supplied coordinates. A 2-channel input is the
        # (eta, phi) points, whose azimuth is PERIODIC: the true particle-particle distance
        # is the shorter arc, so wrap d_phi into [0, pi] before the L2 (a pair straddling the
        # +/-pi seam is adjacent, not ~2*pi apart). Higher-dim deltaR is a dynamic
        # feature-space graph (ordinary Euclidean) -> keep the plain gram L2.
        if x.size(1) == 2:
            eta, phi = x[:, 0, :], x[:, 1, :]
            deta = eta[:, :, None] - eta[:, None, :]
            dphi = (phi[:, :, None] - phi[:, None, :]).abs()
            dphi = torch.minimum(dphi, 2 * torch.pi - dphi)
            pairwise_distance = -(deta ** 2 + dphi ** 2)
        else:
            inner = -2 * torch.matmul(x.transpose(2, 1), x)
            xx = torch.sum(x ** 2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
        if mask is not None:
            eye = torch.eye(num_points, dtype=torch.bool, device=x.device).unsqueeze(0)
            pairwise_distance = pairwise_distance.masked_fill(
                eye | ~mask.bool().unsqueeze(1), float('-inf')
            )
            idx = pairwise_distance.topk(k=k, dim=-1)[1]
        else:
            idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]
    return idx


def gather_neighbors(x, idx):
    """Gather neighbour features. x: (B, C, P), idx: (B, P, K) -> (B, C, P, K)."""
    B, C, P = x.shape
    K = idx.shape[-1]
    idx_base = torch.arange(B, device=x.device).view(-1, 1, 1) * P
    idx_flat = (idx + idx_base).reshape(-1)
    x_flat = x.permute(0, 2, 1).reshape(B * P, C)
    out = x_flat[idx_flat].reshape(B, P, K, C).permute(0, 3, 1, 2)
    return out


class PlainMPNNBlock(nn.Module):
    """Default message-passing block on a (static) kNN graph.

    message m_ij = MLP([h_i, h_j]); aggregate = masked mean over neighbours;
    update h_i' = MLP([h_i, agg]) + shortcut(h_i). All MLPs are 1x1 convs.
    """

    def __init__(self, in_dim, out_dim, in_reps=None):
        super().__init__()
        # in_reps (optional) types the input features for the LLoCa neighbour transport
        # (e.g. "64x0n+16x1n"); its dim must equal in_dim. None -> no transport (the
        # transform is built only when in_reps is given, and used only when forward gets frames).
        self.trafo = None
        if in_reps is not None:
            reps = TensorReps(in_reps)
            assert reps.dim == in_dim, f"in_reps.dim {reps.dim} != in_dim {in_dim}"
            self.trafo = TensorRepsTransform(reps)
        self.message = nn.Sequential(
            nn.Conv2d(2 * in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 1),
            nn.ReLU(),
        )
        self.update = nn.Sequential(
            nn.Conv1d(in_dim + out_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Conv1d(out_dim, out_dim, 1),
        )
        self.act = nn.ReLU()
        if in_dim == out_dim:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_dim, out_dim, 1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_dim)

    def forward(self, features, idx, nbr_mask, frames=None):
        # features: (B, C, P), idx: (B, P, K), nbr_mask: (B, P, K) bool
        B, C, P = features.shape
        K = idx.shape[-1]
        neighbors = gather_neighbors(features, idx)                  # (B, C, P, K)
        # LLoCa: express neighbour j (in its own local frame) in centre i's frame before the
        # message. No-op when frames is None (identity path) or in_reps is scalar-only.
        if frames is not None and self.trafo is not None:
            idx_base = torch.arange(B, device=features.device).view(-1, 1, 1) * P
            idx_flat = (idx + idx_base).reshape(-1)
            neighbors = change_local_frame(neighbors, idx_flat, frames, self.trafo)
        center = features.unsqueeze(-1).expand(-1, -1, -1, K)        # (B, C, P, K)
        m = self.message(torch.cat([center, neighbors], dim=1))     # (B, out, P, K)

        nm = nbr_mask.unsqueeze(1).to(m.dtype)                       # (B, 1, P, K)
        m = m * nm
        count = nm.sum(dim=-1).clamp(min=1.0)                        # (B, 1, P)
        agg = m.sum(dim=-1) / count                                 # (B, out, P): mean

        upd = self.update(torch.cat([features, agg], dim=1))        # (B, out, P)
        sc = self.sc_bn(self.sc(features)) if self.sc is not None else features
        return self.act(sc + upd)


class PlainTransformerBlock(nn.Module):
    """Pre-norm transformer encoder block on torch nn.MultiheadAttention."""

    def __init__(self, embed_dim, num_heads, ffn_ratio=4, dropout=0.1, activation='gelu'):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_ratio * embed_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_ratio * embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, lloca_attn=None):
        # x: (N, 1+P, embed) batch-first. lloca_attn given (learned frames) -> transport q/k/v
        # by reusing this block's MHA weights; None (identity/global) -> plain MHA (bit-identical).
        h = self.norm1(x)
        if lloca_attn is None:
            attn = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        else:
            attn = lloca_transport_attention(
                h, self.attn, lloca_attn, key_padding_mask=key_padding_mask, attn_mask=None,
                dropout_p=self.attn.dropout if self.training else 0.0,
            )
        x = x + self.dropout(attn)
        x = x + self.ffn(self.norm2(x))
        return x


class PlainGraphTrans(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # static plain-MPNN graph backbone
                 gnn_dims=[128, 128, 128],
                 knn_k=16,
                 knn_metric='deltaR',
                 use_fts_bn=True,
                 use_input_concat=True,
                 # transformer
                 embed_dim=128,
                 num_layers=8,
                 num_heads=8,
                 ffn_ratio=4,
                 dropout=0.1,
                 activation='gelu',
                 fc_params=[],
                 # LLoCa tensorial message-passing (purely additive; a no-op for identity/global
                 # frames). attn_reps types the per-head q/k/v transport (embed_dim =
                 # attn_reps.dim * num_heads); hidden_reps_list[i] types MPNN block i's input for
                 # the neighbour transport (None entry -> that block is not transported).
                 attn_reps="8x0n+2x1n",
                 hidden_reps_list=None,
                 # misc
                 for_inference=False,
                 use_amp=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if knn_metric not in ('deltaR', 'minkowski'):
            raise ValueError(f"knn_metric must be 'deltaR' or 'minkowski', got '{knn_metric}'")
        self.knn_k = knn_k
        self.knn_metric = knn_metric
        self.use_fts_bn = use_fts_bn
        self.use_input_concat = use_input_concat
        self.for_inference = for_inference
        self.use_amp = use_amp

        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dim)

        if hidden_reps_list is None:
            hidden_reps_list = [None] * len(gnn_dims)
        assert len(hidden_reps_list) == len(gnn_dims)
        self.gnn_blocks = nn.ModuleList()
        for idx, out_dim in enumerate(gnn_dims):
            in_dim = input_dim if idx == 0 else gnn_dims[idx - 1]
            self.gnn_blocks.append(PlainMPNNBlock(in_dim, out_dim, in_reps=hidden_reps_list[idx]))
        gnn_out = gnn_dims[-1]

        # parameter-free LLoCaAttention for the q/k/v transport (engaged only for learned frames)
        self.lloca_attn = None
        if attn_reps is not None:
            attn_reps_t = TensorReps(attn_reps)
            assert attn_reps_t.dim * num_heads == embed_dim, (
                f"{attn_reps_t.dim}*{num_heads} != embed_dim {embed_dim}"
            )
            self.lloca_attn = LLoCaAttention(attn_reps_t, num_heads)

        bridge_in = gnn_out + input_dim if use_input_concat else gnn_out
        self.bridge = nn.Linear(bridge_in, embed_dim)
        self.bridge_norm = nn.LayerNorm(embed_dim)

        self.blocks = nn.ModuleList([
            PlainTransformerBlock(embed_dim, num_heads, ffn_ratio, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        fcs, in_dim = [], embed_dim
        for out_dim, drop_rate in fc_params:
            fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            in_dim = out_dim
        fcs.append(nn.Linear(in_dim, num_classes))
        self.fc = nn.Sequential(*fcs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
        }

    def forward(self, points, features, v=None, mask=None, frames=None, cls_frames=None):
        '''
        points: (N, 2, P)   features: (N, C, P)   v: (N, 4, P) [px,py,pz,E]   mask: (N, 1, P)
        frames: Frames (N, P, 4, 4) per-particle local frames, or None. The LLoCa transport is
            taken only when frames is given AND not global; identity/global frames take the plain
            path (bit-identical to the non-LLoCa backbone).
        cls_frames: Frames (N, 4, 4) covariant jet frame for the prepended CLS token's slot
            (transport path only); None -> identity slot.
        '''
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)
        else:
            mask = mask.bool()
        features = features * mask
        mask_p = mask.squeeze(1)  # (N, P)

        do_transport = frames is not None and not frames.is_global
        frames_flat = frames.reshape(-1, 4, 4) if do_transport else None

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # static kNN graph (built once, reused by every GNN block)
            if self.knn_metric == 'minkowski' and v is not None:
                idx = knn(v, self.knn_k, metric='minkowski', mask=mask_p)
            else:
                idx = knn(points, self.knn_k, metric='deltaR', mask=mask_p)
            nbr_mask = gather_neighbors(mask.float(), idx).squeeze(1) > 0.5  # (N, P, K)
            # Exclude self. On jets with fewer real constituents than knn_k, topk pads the
            # neighbour list from the tied -inf/self slots, and the realness-only mask above
            # would re-admit the node's OWN index as a "valid" neighbour -- a spurious self
            # message whose value depends on batch-mates and topk tie-breaking. The knn distance
            # already masks self to -inf, so self is never an intended neighbour; drop it here.
            nbr_mask = nbr_mask & (idx != torch.arange(idx.shape[1], device=idx.device)[None, :, None])

            fts = self.bn_fts(features) * mask if self.use_fts_bn else features
            for block in self.gnn_blocks:
                fts = block(fts, idx, nbr_mask, frames=frames_flat) * mask

            if self.use_input_concat:
                bridge_in = torch.cat([fts, features], dim=1)
            else:
                bridge_in = fts

            x = bridge_in.permute(0, 2, 1)  # (N, P, bridge_in)
            x = self.bridge_norm(self.bridge(x))  # (N, P, embed_dim)

            cls = self.cls_token.expand(x.size(0), -1, -1)  # (N, 1, embed_dim)
            x = torch.cat([cls, x], dim=1)  # (N, 1+P, embed_dim)

            # key padding mask (True = ignore); CLS is always valid
            pad = ~mask_p
            key_padding_mask = torch.cat(
                [torch.zeros_like(pad[:, :1]), pad], dim=1
            )  # (N, 1+P)

            # transport path: prepend the CLS frame (covariant jet frame) and prepare the shared,
            # parameter-free LLoCaAttention once for all blocks.
            block_lloca = None
            if do_transport:
                N = features.size(0)
                if cls_frames is not None:
                    cls_mat = cls_frames.matrices.to(frames.dtype)
                else:
                    cls_mat = lorentz_eye((N,), device=frames.device, dtype=frames.dtype)
                seq_mat = torch.cat([cls_mat.unsqueeze(1), frames.matrices], dim=1)  # (N, P+1, 4, 4)
                self.lloca_attn.prepare_frames(Frames(seq_mat, is_global=False, is_identity=False))
                block_lloca = self.lloca_attn

            for block in self.blocks:
                x = block(x, key_padding_mask=key_padding_mask, lloca_attn=block_lloca)

            x_cls = self.norm(x[:, 0])  # (N, embed_dim)
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            return output
