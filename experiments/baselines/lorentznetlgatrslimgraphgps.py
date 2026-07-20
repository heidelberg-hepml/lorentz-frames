"""LorentzNet-L-GATr-slim GraphGPS hybrid: the equivariant slim GraphGPS port.

The slim sibling of CGENNLGATrGraphGPS. It is the geometric-algebra port of the
GraphGPS layer (App. D, Eq. 9-11), but on the *slim* L-GATr representation -- two
streams, ``v`` (Lorentz 4-vectors, (..., V, 4)) and ``s`` (invariant scalars,
(..., S)) -- instead of full 16-component multivectors. This is much cheaper and,
per the L-GATr authors, matches full multivectors on every HEP task they tested.

Each GPS layer fuses a local and a global branch on the (v, s) stream, with every
GraphGPS primitive replaced by its slim equivariant counterpart:

    GraphGPS primitive        slim equivariant replacement (this file)
    -----------------------   --------------------------------------------------
    local MPNN                LorentzNet edge conv (LorentzNetKNNBlock): invariant
                              edge features (|Δp|², <p_i,p_j>) -> SUM scalar /
                              MEAN vector aggregation, with the LorentzNet phi_m gate
    global attention          lgatr.nets.lgatr_slim.SelfAttention (Lorentz attn)
    FFN                       lgatr.nets.lgatr_slim.MLP (gated-linear-unit MLP)
    BatchNorm/LayerNorm       lgatr.nets.lgatr_slim.RMSNorm -- the ONLY norm slim
                              provides (a scale-only, invariant RMS rescale of v+s;
                              there is no slim LayerNorm, and centering vectors would
                              break equivariance)
    Dropout                   lgatr.nets.lgatr_slim.Dropout (shared mask over a
                              vector's 4 components; plain dropout on scalars)
    residual / sum            vector + scalar addition (linear -> equivariant)
    mean-pool readout         mean over real tokens of the invariant scalar stream
                              only (matching pure LorentzNet's h-pool -> graph_dec;
                              the vector geometry already feeds s via each layer's
                              |dp|^2 / <p_i,p_j> edge invariants, so no |v|^2 readout)

The slim SelfAttention / MLP sublayers are used RAW (no internal residual; their
internal qkv-RMSNorm is attention-stability, not a residual norm), so GraphGPS's
external dropout -> residual -> RMSNorm is applied exactly once. The LorentzNet
edge conv carries its own internal residual (like GraphGPS's GatedGCN), so its
branch gets only the external RMSNorm. Equivariant by construction (broken only by
the optional input spurions); the wrapper inherits nn.Module + IdentityFrames.

Reference: LorentzNetLGATrSlimGraphTrans (sequential GNN-stage -> slim-stack); this
interleaves the two per layer. Reuses its LorentzNetKNNBlock / knn / gather helpers.

Input convention (set by the wrapper, channels-first, four-momenta (px, py, pz, E)):
    x:      (B, F, P)   scalar per-particle features
    v:      (B, 4, P)   four-momenta (px, py, pz, E)
    mask:   (B, 1, P)   1 for real particles, 0 for padding
    points: (B, 2, P)   (eta, phi) for the deltaR kNN
"""

import torch
import torch.nn as nn
from lgatr.nets.lgatr_slim import MLP as SlimMLP
from lgatr.nets.lgatr_slim import Dropout as SlimDropout
from lgatr.nets.lgatr_slim import Linear as SlimLinear
from lgatr.nets.lgatr_slim import RMSNorm as SlimRMSNorm
from lgatr.nets.lgatr_slim import SelfAttention as SlimSelfAttention

from experiments.baselines.lorentznetlgatrslimgraphtrans import (
    LorentzNetKNNBlock,
    gather_neighbors,
    knn,
)


class LorentzNetSlimGPSLayer(nn.Module):
    """One equivariant slim GPS layer (Eq. 9-11) on the dual (v, s) stream."""

    def __init__(self, v_channels, s_channels, num_heads, c_weight, use_phi_m,
                 attn_ratio, mlp_ratio, num_layers_mlp, nonlinearity, dropout_prob,
                 n_node_attr=0, attn_dropout=0.0):
        super().__init__()
        # local branch: LorentzNet edge conv (carries its own internal residual).
        # n_node_attr > 0 re-injects the raw input scalars into phi_h every layer,
        # exactly as official LorentzNet's LGEB node_attr (and as the CGENN GPS
        # sibling re-injects its raw node attributes into every local CGENN branch).
        self.gnn = LorentzNetKNNBlock(
            n_h_in=s_channels, n_h_out=s_channels,
            n_v_in=v_channels, n_v_out=v_channels,
            c_weight=c_weight, use_phi_m=use_phi_m,
            n_node_attr=n_node_attr,
        )
        # global branch: raw slim self-attention (no internal residual)
        self.attention = SlimSelfAttention(
            v_channels=v_channels, s_channels=s_channels, num_heads=num_heads,
            attn_ratio=attn_ratio, dropout_prob=None,
        )
        # FFN: raw slim MLP (gated-linear-unit)
        self.mlp = SlimMLP(
            v_channels=v_channels, s_channels=s_channels, nonlinearity=nonlinearity,
            mlp_ratio=mlp_ratio, num_layers=num_layers_mlp, dropout_prob=None,
        )
        # equivariant norm (stateless -> shared) + dropout
        self.norm = SlimRMSNorm()
        self.dropout = SlimDropout(dropout_prob if dropout_prob is not None else 0.0)
        # attention-WEIGHTS dropout: lgatr's sdp_attention forwards dropout_p to one sdpa
        # call over concatenated (mv, s), so one joint mask keeps the streams consistent and
        # the invariant weights preserve equivariance. 0.0 = no-op (kwarg omitted, so non-sdpa
        # backends without dropout_p still work).
        self.attn_dropout = attn_dropout

    def forward(self, v, s, idx, nbr_mask, attn_mask, node_attr=None):
        # v: (B, P, V, 4); s: (B, P, S); slim layers take (vectors, scalars)
        # ---- local branch (Eq. 9): LorentzNet edge conv owns its residual ----
        s_loc, v_loc = self.gnn(s, v, idx, nbr_mask, node_attr=node_attr)  # gnn(h, x) -> (h_new, x_new)
        v_M, s_M = self.norm(v_loc, s_loc)                  # external norm only

        # ---- global branch (Eq. 10): slim Lorentz attention ----
        attn_kwargs = {"attn_mask": attn_mask}
        if self.attn_dropout > 0:
            attn_kwargs["dropout_p"] = self.attn_dropout if self.training else 0.0
        v_a, s_a = self.attention(v, s, **attn_kwargs)
        v_a, s_a = self.dropout(v_a, s_a)
        v_T, s_T = self.norm(v + v_a, s + s_a)

        # ---- fuse by SUM, then slim MLP (Eq. 11) ----
        v_f, s_f = v_M + v_T, s_M + s_T
        v_m, s_m = self.mlp(v_f, s_f)
        v_m, s_m = self.dropout(v_m, s_m)
        v_o, s_o = self.norm(v_f + v_m, s_f + s_m)
        return v_o, s_o


class LorentzNetLGATrSlimGraphGPS(nn.Module):

    def __init__(self,
                 in_s_channels,
                 num_classes,
                 # dual-stream hidden widths
                 hidden_v_channels=16,
                 hidden_s_channels=32,
                 # interleaved GPS layers
                 num_blocks=10,
                 num_heads=8,
                 attn_ratio=1,
                 mlp_ratio=2,
                 num_layers_mlp=2,
                 nonlinearity="gelu",
                 dropout_prob=None,
                 attn_dropout=0.0,
                 # static kNN graph + per-layer LorentzNet edge conv
                 knn_k=16,
                 knn_metric="minkowski",
                 c_weight=1e-3,
                 use_phi_m=True,
                 # official LorentzNet re-injects the raw input scalars into phi_h at
                 # every LGEB layer (node_attr); keep on for faithfulness.
                 use_node_attr=True,
                 # input-stage spurions (break equivariance to the residual symmetry)
                 use_time_spurion=True,
                 use_beam_spurion=True,
                 head_layers=2,
                 **kwargs):
        super().__init__()
        if kwargs:
            # hydra struct mode does not protect against `+model.net.<typo>=x`;
            # the other hybrids raise on unknown keys, so this one must too
            raise TypeError(f"unexpected model kwargs: {sorted(kwargs)}")
        if knn_metric not in ("minkowski", "deltaR"):
            raise ValueError(f"knn_metric must be 'minkowski' or 'deltaR', got {knn_metric!r}")
        self.knn_k = knn_k
        self.knn_metric = knn_metric

        # spurions: hard-coded grade-1 4-vectors in (E, px, py, pz)
        spurions = []
        if use_time_spurion:
            spurions.append([1.0, 0.0, 0.0, 0.0])   # time direction
        if use_beam_spurion:
            spurions.append([0.0, 0.0, 0.0, 1.0])   # beam along z
        self.num_spurions = len(spurions)
        if self.num_spurions > 0:
            self.register_buffer(
                "spurions_4v_buffer", torch.tensor(spurions, dtype=torch.float32),
                persistent=False,
            )

        # equivariant input projection: (1 + num_spurions) vec + scalars -> hidden
        self.linear_in = SlimLinear(
            in_v_channels=1 + self.num_spurions, out_v_channels=hidden_v_channels,
            in_s_channels=in_s_channels, out_s_channels=hidden_s_channels,
        )
        self.use_node_attr = use_node_attr
        self.layers = nn.ModuleList([
            LorentzNetSlimGPSLayer(
                hidden_v_channels, hidden_s_channels, num_heads, c_weight, use_phi_m,
                attn_ratio, mlp_ratio, num_layers_mlp, nonlinearity, dropout_prob,
                n_node_attr=in_s_channels if use_node_attr else 0,
                attn_dropout=attn_dropout,
            )
            for _ in range(num_blocks)
        ])

        # invariant head: pooled scalars -> logits. SCALAR-ONLY, matching pure LorentzNet
        # (mean-pool of h -> graph_dec). Its LGEB already funnels the four-vector geometry
        # into the scalars via per-layer |dp|^2 / <p_i,p_j> edge invariants (as does the local
        # branch here), so we do NOT add |v|^2 self-norms -- that would make the head richer
        # than the GNN it mirrors. Contrast CGENN GPS, whose pure reference IS rich.
        head, d = [], hidden_s_channels
        for _ in range(head_layers):
            nd = max(d // 2, num_classes)
            head += [nn.Linear(d, nd), nn.GELU()]
            d = nd
        head += [nn.Linear(d, num_classes)]
        self.head = nn.Sequential(*head)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, v, mask, points=None):
        # x: (B, F, P); v: (B, 4, P) (px, py, pz, E); mask: (B, 1, P); points: (B, 2, P)
        s_in = x.transpose(1, 2)                                  # (B, P, F)
        mask_b = mask.transpose(1, 2).bool().squeeze(-1)         # (B, P)
        B, P, _ = s_in.shape

        # static kNN graph (computed once, reused by every layer's local branch)
        if self.knn_metric == "minkowski":
            idx = knn(v, self.knn_k, metric="minkowski", mask=mask_b)
        else:
            if points is None:
                raise ValueError("knn_metric='deltaR' requires `points`")
            idx = knn(points, self.knn_k, metric="deltaR", mask=mask_b)
        nbr_valid = gather_neighbors(mask_b.unsqueeze(-1), idx).squeeze(-1)
        nbr_mask = nbr_valid & mask_b.unsqueeze(-1)              # (B, P, K)
        # Exclude self (see the GraphTrans sibling): sparse jets (n_real < knn_k) let topk fill
        # the neighbour list from the tied -inf/self slots, and the realness masks above would
        # re-admit the node's own index as a spurious self message.
        nbr_mask = nbr_mask & (idx != torch.arange(idx.shape[1], device=idx.device)[None, :, None])

        # particle 4-vector (reordered to time-first) + spurion channels
        v_etxyz = v.transpose(1, 2)[..., [3, 0, 1, 2]]           # (B, P, 4) (E, px, py, pz)
        x_vec = v_etxyz.unsqueeze(2)                             # (B, P, 1, 4)
        if self.num_spurions > 0:
            spur = self.spurions_4v_buffer.to(x_vec.dtype)
            x_vec = torch.cat([x_vec, spur[None, None].expand(B, P, -1, -1)], dim=2)

        # equivariant input projection, then interleaved GPS layers. The raw input
        # scalars ride along as per-layer node attributes (official LorentzNet's
        # node_attr), matching the CGENN GPS sibling's raw re-injection.
        v_h, s_h = self.linear_in(x_vec, s_in)                  # (B,P,V,4), (B,P,S)
        node_attr = s_in if self.use_node_attr else None
        attn_mask = mask_b[:, None, None, :]                    # (B, 1, 1, P) bool, True = real
        for layer in self.layers:
            v_h, s_h = layer(v_h, s_h, idx, nbr_mask, attn_mask, node_attr=node_attr)

        # masked mean pool of the SCALAR stream only (matching pure LorentzNet's h-pool). The
        # vector stream v_h fed s_h through every layer's invariant edge features, so its content
        # is already in s_h; its final per-node self-norm is intentionally not read out.
        m = mask_b[..., None].to(s_h.dtype)                     # (B, P, 1)
        denom = m.sum(dim=1).clamp(min=1.0)                     # (B, 1)
        s_pool = (s_h * m).sum(dim=1) / denom                  # (B, S)
        return self.head(s_pool)
