"""CGENN-L-GATr GraphGPS hybrid: the equivariant GraphGPS port (MV + scalar streams).

The equivariant member of the GraphGPS family. It is the geometric-algebra port of
the GraphGPS layer (paper App. D, Eq. 9-11): each layer runs a local and a global
branch on the SAME dual stream -- ``mv`` (multivectors, (..., C, 16)) and ``s``
(auxiliary scalars, (..., C)) -- and fuses them, but every GraphGPS primitive is
replaced by its equivariant counterpart so the whole model is Lorentz-equivariant
by construction (symmetry broken only by the optional input spurions):

    GraphGPS primitive        equivariant replacement (this file)
    -----------------------   --------------------------------------------------
    local MPNN                CGENN message passing (CGLayer): geometric-product
                              messages on mv + invariant-gated scalar messages, with the
                              same static relative-momentum edge multivectors
                              [p_i-p_j, raw_i, raw_j] the GraphTrans cousin injects
                              (use_explicit_edge_features, on by default)
    global attention          L-GATr SelfAttention (geometric multi-head attn)
    FFN (2-layer MLP)         GeoMLP -- starts with a GeometricBilinear (the
                              geometric product), then gated EquiLinear layers
    BatchNorm/LayerNorm       EquiLayerNorm (GA-norm rescale on mv, LayerNorm on s)
    Dropout                   GradeDropout (per-grade on mv, plain on s)
    residual adds (x + ...)   multivector + scalar addition (linear -> equivariant)
    sum fusion (X_M + X_T)    multivector + scalar addition
    mean-pool readout         mean over real tokens (equivariant), then
                              get_invariants(mv) (grade-0 + norms of grades 1-4) + pooled scalars

The L-GATr attention / GeoMLP sublayers are used RAW -- their internal
dropout/residual/norm are disabled (``dropout_prob=None``; the LGATrBlock's
pre-norm + residual are not used) -- so GraphGPS's external dropout -> residual ->
norm is applied exactly once (avoiding the double-residual trap of dropping a whole
LGATrBlock into a branch). EquiLayerNorm is stateless, so one instance is shared.

Unlike the non-equivariant family members this uses NO BatchNorm (it would break
equivariance over multivector components) and NO class token (mean-pool readout).
Equivariant by construction, so the wrapper inherits nn.Module with IdentityFrames
(no LLoCa canonicalization), like CGENNLGATrGraphTrans.

Reference: CGENNLGATrGraphTrans (the sequential CGENN-stage -> L-GATr-stack version)
-- this interleaves the two per layer. Reuses its CliffordAlgebra / CGLayer /
generate_edges_vectorized for the local branch and graph.

Input convention (set by the wrapper, four-momenta time-first as (E, px, py, pz)):
    x:      (B, P, C)   scalar per-particle features
    v:      (B, P, 4)   four-momenta (E, px, py, pz), embedded as multivectors
    mask:   (B, P)      1 for real particles, 0 for padding
    points: (B, P, 2)   (eta, phi) for the deltaR kNN
"""

from dataclasses import replace

import torch
import torch.nn as nn
from lgatr import embed_vector, get_num_spurions, get_spurions
from lgatr.layers.attention.config import SelfAttentionConfig
from lgatr.layers.attention.self_attention import SelfAttention
from lgatr.layers.dropout import GradeDropout
from lgatr.layers.layer_norm import EquiLayerNorm
from lgatr.layers.linear import EquiLinear
from lgatr.layers.mlp.config import MLPConfig
from lgatr.layers.mlp.mlp import GeoMLP

from experiments.baselines.CGENNLGATrGraphTransHybrid import (
    CGLayer,
    CliffordAlgebra,
    generate_edges_vectorized,
    get_invariants,
)


class CGENNLGATrGPSLayer(nn.Module):
    """One equivariant GPS layer (Eq. 9-11) on the dual (mv, s) stream."""

    def __init__(self, algebra, mv_channels, s_channels, num_heads,
                 cgenn_aggregation, cgenn_layer_type, cgenn_normalization_init,
                 increase_hidden_channels_attention, increase_hidden_channels_mlp,
                 num_hidden_layers_mlp, head_scale, multi_query, activation, dropout_prob,
                 edge_attr_x_dim=0, node_attr_x_dim=0, node_attr_h_dim=0):
        super().__init__()
        # ---- local branch: one CGENN message-passing layer. residual=False because the GPS
        #      layer owns the external residual. edge_attr_x_dim > 0 -> the layer consumes the
        #      static relative-momentum edge multivectors; node_attr_{x,h}_dim > 0 -> the raw
        #      inputs are re-injected as per-node attributes every layer (theta_x/theta_h) --
        #      all three matching the GraphTrans CGENN stage. The static features are passed in
        #      forward and shared across layers.
        self.cgenn = CGLayer(
            algebra,
            mv_channels, mv_channels, mv_channels,
            s_channels, s_channels, s_channels,
            edge_attr_x=edge_attr_x_dim, edge_attr_h=0,
            node_attr_x=node_attr_x_dim, node_attr_h=node_attr_h_dim,
            aggregation=cgenn_aggregation, use_invariants_to_update=True,
            residual=False, normalization_init=cgenn_normalization_init,
            layer_type=cgenn_layer_type,
        )
        # ---- global branch: raw L-GATr geometric self-attention (no internal
        #      dropout -> the GPS layer applies the external dropout) ----
        attn_cfg = replace(
            SelfAttentionConfig(num_heads=num_heads, multi_query=multi_query,
                                increase_hidden_channels=increase_hidden_channels_attention,
                                head_scale=head_scale),
            in_mv_channels=mv_channels, out_mv_channels=mv_channels,
            in_s_channels=s_channels, out_s_channels=s_channels,
            output_init="small", dropout_prob=None,
        )
        self.attention = SelfAttention(attn_cfg)
        # ---- FFN: raw GeoMLP (geometric product first) ----
        mlp_cfg = replace(
            MLPConfig(activation=activation,
                      increase_hidden_channels=increase_hidden_channels_mlp,
                      num_hidden_layers=num_hidden_layers_mlp),
            mv_channels=mv_channels, s_channels=s_channels, dropout_prob=None,
        )
        self.mlp = GeoMLP(mlp_cfg)
        # ---- equivariant norm (stateless -> shared) + dropout ----
        self.norm = EquiLayerNorm()
        self.dropout = GradeDropout(dropout_prob if dropout_prob is not None else 0.0)

    def forward(self, mv, s, edges, attn_mask, edge_attr_x=None,
                node_attr_x=None, node_attr_h=None):
        # mv: (B, P, C_mv, 16); s: (B, P, C_s); edge_attr_x: (num_edges, E, 16) static
        # relative-momentum edge multivectors; node_attr_x/node_attr_h: the static raw mv /
        # scalar inputs re-injected per node (or None when explicit edge features are off).
        B, P = mv.shape[0], mv.shape[1]

        # ---- local branch (Eq. 9): CGENN message passing on the static kNN graph.
        # The dense B*P layout (padded slots included) is kept deliberately: it
        # matches official CGENN, whose theta_h BatchNorm also runs over the padded
        # nodes. Edges connect real nodes only, so padding does not propagate
        # through message passing -- only into the scalar BN statistics, as upstream.
        s_loc, mv_loc = self.cgenn(
            s.reshape(B * P, -1), mv.reshape(B * P, -1, 16), edges,
            node_attr_h=node_attr_h, node_attr_x=node_attr_x,
            edge_attr_h=None, edge_attr_x=edge_attr_x,
        )
        mv_loc = mv_loc.view(B, P, -1, 16)
        s_loc = s_loc.view(B, P, -1)
        mv_loc, s_loc = self.dropout(mv_loc, s_loc)
        mv_M, s_M = self.norm(mv + mv_loc, scalars=s + s_loc)

        # ---- global branch (Eq. 10): L-GATr geometric attention ----
        mv_att, s_att = self.attention(mv, scalars=s, attn_mask=attn_mask)
        mv_att, s_att = self.dropout(mv_att, s_att)
        mv_T, s_T = self.norm(mv + mv_att, scalars=s + s_att)

        # ---- fuse by SUM, then GeoMLP (Eq. 11) ----
        mv_f, s_f = mv_M + mv_T, s_M + s_T
        mv_ff, s_ff = self.mlp(mv_f, scalars=s_f)
        mv_ff, s_ff = self.dropout(mv_ff, s_ff)
        mv_out, s_out = self.norm(mv_f + mv_ff, scalars=s_f + s_ff)
        return mv_out, s_out


class CGENNLGATrGraphGPS(nn.Module):

    def __init__(self,
                 in_s_channels: int,
                 hidden_mv_channels: int,
                 hidden_s_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 num_heads: int,
                 k: int = None,
                 knn_metric: str = "deltaR",
                 # per-layer CGENN message passing
                 cgenn_aggregation: str = "mean",
                 cgenn_layer_type: str = "fc",
                 cgenn_normalization_init=None,  # official CGENN: no NormalizationLayer (tag_cgenn row)
                 # relative-momentum edge features for the local CGENN branch (as in the
                 # GraphTrans cousin's CGENN stage). Computed once from the raw four-momenta
                 # and shared across layers. True -> like-for-like local MPNN; False -> the
                 # earlier bare message passing (hidden stream only).
                 use_explicit_edge_features: bool = True,
                 # input spurions (break equivariance to the residual symmetry)
                 beam_spurion: str = "xyplane",
                 add_time_spurion: bool = True,
                 beam_mirror: bool = True,
                 # L-GATr attention / MLP hyperparameters
                 activation: str = "gelu",
                 multi_query: bool = False,
                 increase_hidden_channels_attention: int = 2,
                 increase_hidden_channels_mlp: int = 2,
                 num_hidden_layers_mlp: int = 1,
                 head_scale: bool = False,
                 dropout_prob: float = None,
                 head_layers: int = 2,
                 **kwargs):
        super().__init__()
        if knn_metric not in ("deltaR", "minkowski"):
            raise ValueError(f"knn_metric must be 'deltaR' or 'minkowski', got '{knn_metric}'")
        self.algebra = CliffordAlgebra((1.0, -1.0, -1.0, -1.0))
        self.k = k
        self.knn_metric = knn_metric
        self.spurion_kwargs = dict(
            beam_spurion=beam_spurion, add_time_spurion=add_time_spurion, beam_mirror=beam_mirror
        )
        self.num_spurions = get_num_spurions(beam_spurion, add_time_spurion, beam_mirror=beam_mirror)

        # input embedding: (1 particle + num_spurions) mv channels + scalars -> hidden
        self.linear_in = EquiLinear(
            in_mv_channels=1 + self.num_spurions, out_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels, out_s_channels=hidden_s_channels,
        )
        # static relative-momentum edge features = [p_i - p_j, raw_i, raw_j] over the raw
        # (1 + num_spurions) input multivector channels -> 1 + 2*(1+num_spurions) mv channels,
        # matching the GraphTrans CGENN stage. 0 disables them (bare hidden-stream message passing).
        self.use_explicit_edge_features = use_explicit_edge_features
        raw_mv_channels = 1 + self.num_spurions
        # Matching the GraphTrans CGENN stage, the explicit-edge-feature path injects THREE static
        # signals into every local CGENN layer: the relative-momentum edge multivectors, and the
        # raw mv / raw scalar inputs re-injected as per-node attributes (theta_x/theta_h). 0 = off.
        edge_attr_x_dim = (1 + 2 * raw_mv_channels) if use_explicit_edge_features else 0
        node_attr_x_dim = raw_mv_channels if use_explicit_edge_features else 0
        node_attr_h_dim = in_s_channels if use_explicit_edge_features else 0
        self.layers = nn.ModuleList([
            CGENNLGATrGPSLayer(
                self.algebra, hidden_mv_channels, hidden_s_channels, num_heads,
                cgenn_aggregation, cgenn_layer_type, cgenn_normalization_init,
                increase_hidden_channels_attention, increase_hidden_channels_mlp,
                num_hidden_layers_mlp, head_scale, multi_query, activation, dropout_prob,
                edge_attr_x_dim=edge_attr_x_dim,
                node_attr_x_dim=node_attr_x_dim, node_attr_h_dim=node_attr_h_dim,
            )
            for _ in range(num_blocks)
        ])
        self.final_norm = EquiLayerNorm()

        # invariant head: the FULL multivector invariants (grade-0 plus the norms of grades
        # 1-4) of the pooled mv, concatenated with the pooled scalars, then MLP -> logits.
        # This is exactly pure CGENN's readout (`get_invariants` = grade-0 + qs(grades[1:])).
        # We extract every grade's invariant rather than only extract_scalar's grade-0 because
        # the mean-pool cannot learn to route higher-grade content into grade-0 the way a CLS
        # token can (see the LorentzNet-slim GPS head, which likewise pools vector norms) --
        # so the discarded-grade invariants (vector/trivector norms, the two bivector
        # invariants, the pseudoscalar) are surfaced here. n_mv_inv = len(grades) invariants
        # per mv channel; all are Lorentz-invariant, so the readout stays invariant.
        n_mv_inv = len(self.algebra.grades)
        head, d = [], hidden_mv_channels * n_mv_inv + hidden_s_channels
        for _ in range(head_layers):
            nd = max(d // 2, num_classes)
            head += [nn.Linear(d, nd), nn.GELU()]
            d = nd
        head += [nn.Linear(d, num_classes)]
        self.head = nn.Sequential(*head)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, v, mask, points):
        # x: (B, P, C_s); v: (B, P, 4) [E, px, py, pz]; mask: (B, P); points: (B, P, 2)
        B, P, _ = x.shape
        device = x.device

        # Stage 1: embed four-momenta as multivectors; spurions as extra mv channels
        mv = embed_vector(v[:, :, None, :])              # (B, P, 1, 16)
        if self.num_spurions > 0:
            spur = get_spurions(**self.spurion_kwargs).to(device=device, dtype=x.dtype)
            mv = torch.cat([mv, spur[None, None].expand(B, P, -1, -1)], dim=2)
        s = x

        # Stage 2: static kNN graph (native dtype; shared by every layer's local branch)
        fourmomenta_flat = v if (self.knn_metric == "minkowski" and self.k is not None) else None
        edges = generate_edges_vectorized(
            mask, points, self.k, P, device,
            metric=self.knn_metric, fourmomenta=fourmomenta_flat,
        )

        # Stage 2b: static relative-momentum edge features from the RAW input multivectors
        # (before linear_in), [p_i - p_j, raw_i, raw_j], shared across every layer's local
        # CGENN branch -- the same geometric edge signal the GraphTrans CGENN stage injects.
        edge_attr_x = node_attr_x = node_attr_h = None
        if self.use_explicit_edge_features:
            i, j = edges
            mv_raw = mv.reshape(B * P, -1, 16)            # (B*P, 1+num_spurions, 16)
            particle_diff = mv_raw[i, :1] - mv_raw[j, :1]  # (num_edges, 1, 16)
            edge_attr_x = torch.cat([particle_diff, mv_raw[i], mv_raw[j]], dim=1)
            # raw inputs re-injected as per-node attributes every layer (as in GraphTrans)
            node_attr_x = mv_raw                          # (B*P, 1+num_spurions, 16) raw mv
            node_attr_h = s.reshape(B * P, -1)            # (B*P, in_s_channels) raw scalars

        # Stage 3: equivariant input projection to hidden channels
        mv, s = self.linear_in(mv, scalars=s)            # (B, P, C_mv, 16), (B, P, C_s)

        # Stage 4: interleaved equivariant GPS layers
        attn_mask = mask[:, None, None, :]               # (B, 1, 1, P) bool, True = real
        for layer in self.layers:
            mv, s = layer(mv, s, edges, attn_mask, edge_attr_x=edge_attr_x,
                          node_attr_x=node_attr_x, node_attr_h=node_attr_h)

        # Stage 5: final norm + per-node invariants + masked mean pool + head.
        # get_invariants is applied PER NODE and THEN mean-pooled (extract-then-pool),
        # exactly as pure CGENN's readout (cgenn.py) and the LorentzNet-slim GPS head --
        # NOT pool-then-extract. Grade-0 agrees either way (linear), but the grade norms
        # are nonlinear, so the mean of the per-node norms (this) differs from the norm of
        # the mean multivector.
        mv, s = self.final_norm(mv, scalars=s)
        m = mask[..., None].to(s.dtype)                  # (B, P, 1)
       
        denom = m.sum(dim=1).clamp(min=1.0)              # (B, 1): real count, NOT padded batch-max and thus batch invariant unlike the official CGENN quirk (just made a choice to go for this)
        inv_nodes = get_invariants(self.algebra, mv).flatten(2)      # (B, P, C_mv * n_mv_inv)
        inv = (inv_nodes * m).sum(dim=1) / denom                     # (B, C_mv * n_mv_inv)
        s_pool = (s * m).sum(dim=1) / denom                          # (B, C_s)
        return self.head(torch.cat([inv, s_pool], dim=-1))
