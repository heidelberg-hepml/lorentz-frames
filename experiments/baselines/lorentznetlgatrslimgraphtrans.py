"""Static-kNN LorentzNet -> L-GATr-slim hybrid (per-edge attention gate on by default).

The internally-equivariant analogue of ParticleNetParTGraphTrans: a LorentzNet
edge convolution on a static Minkowski-kNN graph feeds an L-GATr-slim transformer.
Both stages are Lorentz-equivariant, and symmetry is broken only by input-stage
spurions, so the model runs on identity frames (its wrapper inherits nn.Module,
like CGENNLGATrGraphTransWrapper).

Three deliberate departures from canonical LorentzNet:

  1. Per-edge soft-attention gate (LorentzNet's phi_m sigmoid) is OPTIONAL
     (use_phi_m, on by default). Aggregation matches official LorentzNet:
     SUM for the scalar stream, MEAN for the vector stream. Turning phi_m off
     gives a gateless block, leaving soft attention to the transformer
     downstream -- the GraphGPS division of labour.

  2. STATIC kNN graph, not ParticleNet-style dynamic EdgeConv. The graph is
     built ONCE per forward pass and reused across every block. Metric:
       * 'minkowski' (default) -- Lorentz-invariant interval; equivariant.
       * 'deltaR'              -- L2 on (eta, phi); not Lorentz invariant,
         for ablation / parity with ParticleNet's first layer.

  3. Spurions at the GNN INPUT ONLY, as pure grade-1 4-vectors, toggled by two
     booleans (use_time_spurion -> [E=1,0,0,0]; use_beam_spurion -> [0,0,0,pz=1]).
     With both on (default) the residual symmetry is SO(2) azimuthal rotations
     about the beam. They are broadcast as extra vector channels on every
     particle; they are NEVER graph nodes.

The transformer is L-GATr-SLIM (scalars + 4-vectors only). Class logits come
directly from LGATrSlim's output scalar channels, which are Lorentz invariants
under the residual subgroup the spurions preserve.

Internal 4-vector ordering: (E, px, py, pz), time at index 0 (lgatr's grade-1
convention). The `knn` helper natively expects (px, py, pz, E) for the minkowski
branch, so kNN is computed BEFORE the internal reorder; everything downstream
uses (E, px, py, pz).

Adapted from the weaver source: the weaver tagger wrappers, ``get_model`` and
``get_loss`` are removed (the model returns logits directly and is driven by
``experiments.tagging.wrappers.LorentzNetLGATrSlimGraphTransWrapper``).
"""

import torch
from torch import nn

from lgatr import LGATrSlim


# Minkowski math (mostly-minus signature, time at index 0 in (E, px, py, pz))


def normsq4(p):
    r""":math:`\|p\|^2 = p_0^2 - p_1^2 - p_2^2 - p_3^2` along the last dim.
    Assumes (E, px, py, pz) ordering with time at index 0."""
    psq = p.pow(2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def dotsq4(p, q):
    r""":math:`\langle p, q\rangle = p_0 q_0 - p_1 q_1 - p_2 q_2 - p_3 q_3`.
    Assumes (E, px, py, pz) ordering with time at index 0."""
    psq = p * q
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def psi(p):
    r""":math:`\psi(p) = \mathrm{sgn}(p)\log(|p|+1)`. Compresses dynamic
    range of edge invariants, which span many orders of magnitude in HEP."""
    return torch.sign(p) * torch.log(p.abs() + 1)


# kNN with a Lorentz-invariant or deltaR metric. Same robust kNN as
# ParticleNetParTGraphTrans: k capped at (P-1) so it never crashes on sparse events;
# an optional mask keeps padded slots out of the neighbour search.


def knn(x, k, metric='deltaR', mask=None):
    num_points = x.size(-1)
    k = min(k, max(1, num_points - 1))
    if metric == 'minkowski':
        # x: (N, 4, P) in (px, py, pz, E). Rank by |(x_i - x_j)^2|_minkowski.
        sig = x.new_tensor([-1.0, -1.0, -1.0, 1.0]).view(1, -1, 1)
        xp = x * sig
        msq = (x * xp).sum(dim=1, keepdim=True)  # (N, 1, P): E^2 - p^2
        gram = torch.matmul(x.transpose(2, 1), xp)  # (N, P, P): <x_i, x_j>_mink
        d2 = msq.transpose(2, 1) + msq - 2 * gram  # (N, P, P)
        pairwise_distance = -d2.abs()
        if mask is not None:
            # padded senders large-but-finite, self -inf: on sparse jets (n_real<=k) topk
            # fills surplus slots with padding, never self (nbr_mask rejects both; this just
            # makes the fill deterministic, matching the ParticleNetParT helper).
            pairwise_distance = pairwise_distance.masked_fill(~mask.bool().unsqueeze(1), -1e30)
        eye = torch.eye(num_points, dtype=torch.bool, device=x.device).unsqueeze(0)
        pairwise_distance = pairwise_distance.masked_fill(eye, float('-inf'))
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
    else:
        # 'deltaR': squared-L2 on the coords. A 2-channel input is (eta, phi); azimuth is
        # PERIODIC, so wrap d_phi into [0, pi] before the L2 (a pair across the +/-pi seam is
        # adjacent, not ~2pi apart). Higher-dim deltaR is Euclidean -> plain gram L2.
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
            # padded senders large-but-finite, self -inf (see the minkowski branch)
            eye = torch.eye(num_points, dtype=torch.bool, device=x.device).unsqueeze(0)
            pairwise_distance = pairwise_distance.masked_fill(~mask.bool().unsqueeze(1), -1e30)
            pairwise_distance = pairwise_distance.masked_fill(eye, float('-inf'))
            idx = pairwise_distance.topk(k=k, dim=-1)[1]
        else:
            idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]
    return idx


# Dense neighbour gather (ParticleNet style)


def gather_neighbors(x, idx):
    """Gather per-edge features into a dense (B, P, K, ...) tensor.

    Parameters
    ----------
    x : (B, P, *feat) Tensor
    idx : (B, P, K) long Tensor

    Returns
    -------
    (B, P, K, *feat) Tensor
    """
    B, P = x.shape[:2]
    K = idx.shape[2]
    feat_shape = x.shape[2:]

    x_flat = x.reshape(B * P, *feat_shape)
    offset = (torch.arange(B, device=x.device) * P).view(B, 1, 1)
    idx_global = (idx + offset).reshape(-1)
    gathered = x_flat[idx_global]
    return gathered.view(B, P, K, *feat_shape)


# LorentzNet-style edge conv on a static kNN graph (NO soft-attention gate)


class LorentzNetKNNBlock(nn.Module):
    """Lorentz-equivariant edge convolution on a static kNN graph.

    Mirrors LorentzNet's LGEB: the optional per-edge soft-attention gate (phi_m,
    on by default) weights each message; the scalar stream is then SUM-aggregated
    and the vector stream MEAN-aggregated over valid neighbours -- exactly as in
    official LorentzNet (unsorted_segment_sum / unsorted_segment_mean). Set
    use_phi_m=False for a gateless plain-sum/mean block (GraphGPS-style, leaving
    soft attention to the transformer downstream).

    Streams maintained:
        h : (B, P, n_h_in)        Lorentz-invariant scalars
        x : (B, P, n_v_in, 4)     Lorentz-equivariant 4-vectors
                                  in (E, px, py, pz) ordering
    """

    def __init__(self, n_h_in, n_h_out, n_v_in, n_v_out, c_weight=1e-3, use_phi_m=True,
                 n_node_attr=0):
        super().__init__()
        self.n_v_in = n_v_in
        self.n_v_out = n_v_out
        self.c_weight = c_weight
        # official LorentzNet re-injects the RAW input scalars as per-node attributes
        # into phi_h at EVERY layer (LGEB's n_node_attr / node_attr); n_node_attr=0
        # disables the pathway (the pre-audit behaviour).
        self.n_node_attr = n_node_attr

        # 2 * n_v_out invariants per edge (psi-compressed Minkowski
        # norm-of-diff and dot-product, one per projected vector channel).
        n_edge_inv = 2 * n_v_out
        # phi_e input: [h_i, h_j - h_i, edge_invariants]
        n_edge_in = 2 * n_h_in + n_edge_inv

        # Edge MLP (Conv2d on (B, C, P, K) == per-edge MLP).
        self.phi_e = nn.Sequential(
            nn.Conv2d(n_edge_in, n_h_out, 1, bias=False),
            nn.BatchNorm2d(n_h_out),
            nn.ReLU(),
            nn.Conv2d(n_h_out, n_h_out, 1),
            nn.ReLU(),
        )
        # LorentzNet's phi_m: a learned per-edge sigmoid gate in [0,1] weighting each
        # message before aggregation (scalar and vector). It weights within the kNN
        # neighbourhood, complementing the global transformer. Toggle off for the pure
        # GraphGPS division of labour (transformer-only attention).
        self.phi_m = (
            nn.Sequential(nn.Conv2d(n_h_out, 1, 1), nn.Sigmoid()) if use_phi_m else None
        )

        # Scalar node update (LorentzNet-style). Input: [h, aggregated message(, node_attr)]
        # -- the optional node_attr slot is official LGEB's per-layer raw-scalar re-injection.
        self.phi_h = nn.Sequential(
            nn.Conv1d(n_h_in + n_h_out + n_node_attr, n_h_out, 1),
            nn.BatchNorm1d(n_h_out),
            nn.ReLU(),
            nn.Conv1d(n_h_out, n_h_out, 1),
        )

        # Lorentz-equivariant linear mix across vector channels
        # (channel-axis only -> commutes with Lambda on the 4-vector axis).
        self.vec_mix = nn.Linear(n_v_in, n_v_out, bias=False)

        # phi_x: per-edge, per-channel weight for the vector update.
        coord_layer = nn.Conv2d(n_h_out, n_v_out, 1, bias=False)
        nn.init.xavier_uniform_(coord_layer.weight, gain=0.001)
        self.phi_x = nn.Sequential(
            nn.Conv2d(n_h_out, n_h_out, 1),
            nn.ReLU(),
            coord_layer,
        )

        self.residual_h = (n_h_in == n_h_out)

    def forward(self, h, x, idx, neighbor_mask, node_attr=None):
        """
        Parameters
        ----------
        h : (B, P, n_h_in)
        x : (B, P, n_v_in, 4)
        idx : (B, P, K) -- neighbour indices (static across blocks)
        neighbor_mask : (B, P, K) bool, True where edge is valid.
        node_attr : (B, P, n_node_attr) or None -- the RAW input scalars, re-injected
            into phi_h every layer exactly as official LorentzNet's LGEB node_attr.
        """
        B, P, K = idx.shape

        # 1. Equivariant linear mix of input vector channels.
        x_proj = torch.einsum("bpcd,oc->bpod", x, self.vec_mix.weight)

        # 2. Gather neighbour features.
        h_j = gather_neighbors(h, idx)             # (B, P, K, n_h_in)
        x_j = gather_neighbors(x_proj, idx)        # (B, P, K, n_v_out, 4)
        h_i_exp = h.unsqueeze(2).expand(-1, -1, K, -1)
        x_i = x_proj.unsqueeze(2)                  # broadcasts over K

        # 3. Lorentz-invariant edge features.
        # When particle channels are mixed with spurion channels at the
        # input, the dot-product edge features pick up particle-spurion
        # cross terms automatically.
        x_diff = x_i - x_j                                          # (B, P, K, n_v_out, 4)
        norms = normsq4(x_diff)                                     # (B, P, K, n_v_out)
        dots = dotsq4(x_i, x_j)                                     # broadcasts over K
        edge_inv = psi(torch.cat([norms, dots], dim=-1))           # (B, P, K, 2*n_v_out)

        # 4. Per-edge MLP.
        edge_feat = torch.cat([h_i_exp, h_j - h_i_exp, edge_inv], dim=-1)
        edge_feat = edge_feat.permute(0, 3, 1, 2).contiguous()      # (B, C, P, K)
        m = self.phi_e(edge_feat)                                   # (B, n_h_out, P, K)
        if self.phi_m is not None:
            # LorentzNet edge gate: weight each message (feeds both updates below)
            m = m * self.phi_m(m)

        # 5. Mask invalid edges, then aggregate. Matching official LorentzNet:
        #    the scalar stream uses SUM (unsorted_segment_sum), the vector stream
        #    uses MEAN (unsorted_segment_mean).
        nm = neighbor_mask.unsqueeze(1).to(m.dtype)                 # (B, 1, P, K)
        m = m * nm
        nm_count = nm.sum(dim=-1).clamp(min=1.0)                    # (B, 1, P)
        h_msg = m.sum(dim=-1)                                       # (B, n_h_out, P): SUM

        # 6. Scalar update: [h, agg(, node_attr)] as in official LGEB's h_model.
        h_in = h.transpose(1, 2)                                    # (B, n_h_in, P)
        phi_h_in = [h_in, h_msg]
        if self.n_node_attr > 0:
            assert node_attr is not None, "block built with n_node_attr > 0 but got node_attr=None"
            phi_h_in.append(node_attr.transpose(1, 2))              # (B, n_node_attr, P)
        h_update = self.phi_h(torch.cat(phi_h_in, dim=1))          # (B, n_h_out, P)
        h_new = h_update.transpose(1, 2)
        if self.residual_h:
            h_new = h_new + h

        # 7. Equivariant vector update.
        coord_w = self.phi_x(m)                                     # (B, n_v_out, P, K)
        coord_w = coord_w.permute(0, 2, 3, 1).unsqueeze(-1)        # (B, P, K, n_v_out, 1)
        trans = (x_diff * coord_w).clamp(min=-100, max=100)        # LorentzNet stability
        trans = trans * neighbor_mask.unsqueeze(-1).unsqueeze(-1).to(trans.dtype)
        x_msg = trans.sum(dim=2) / nm_count.transpose(1, 2).unsqueeze(-1)
        x_new = x_proj + self.c_weight * x_msg

        return h_new, x_new


# Hybrid model


class LorentzNetLGATrSlimGraphTrans(nn.Module):
    """Static-kNN LorentzNet (phi_m gate on by default) -> L-GATr-slim hybrid with
    boolean-toggled input-stage spurions and a learnable scalar CLS."""

    def __init__(
        self,
        in_s_channels,
        num_classes,
        # GNN stage
        n_gnn_layers=4,
        n_h_hidden=72,
        n_v_hidden=16,
        knn_k=16,
        knn_metric="minkowski",
        c_weight=1e-3,
        use_phi_m=True,
        # official LorentzNet re-injects the raw input scalars into phi_h at every
        # LGEB layer (node_attr); keep on for faithfulness, off = pre-audit ablation.
        use_node_attr=True,
        # Bridge
        concat_original=True,
        # Symmetry breaking (input-stage only): time + beam reference 4-vectors.
        use_time_spurion=True,
        use_beam_spurion=True,
        # Global (CLS) token
        global_token=True,
        # L-GATr-slim
        hidden_v_channels=16,
        hidden_s_channels=32,
        num_blocks=10,
        num_heads=8,
        mlp_ratio=2,
        attn_ratio=1,
        num_layers_mlp=2,
    ):
        super().__init__()
        if knn_metric not in ("minkowski", "deltaR"):
            raise ValueError(
                f"knn_metric must be 'minkowski' or 'deltaR', got {knn_metric!r}"
            )

        self.in_s_channels = in_s_channels
        self.n_h_hidden = n_h_hidden
        self.n_v_hidden = n_v_hidden
        self.knn_k = knn_k
        self.knn_metric = knn_metric
        self.concat_original = concat_original
        self.global_token = global_token
        self.use_time_spurion = use_time_spurion
        self.use_beam_spurion = use_beam_spurion
        self.use_node_attr = use_node_attr

        # ---- Spurions: hard-coded grade-1 4-vectors in (E, px, py, pz).
        spurions = []
        if use_time_spurion:
            spurions.append([1.0, 0.0, 0.0, 0.0])   # time direction
        if use_beam_spurion:
            spurions.append([0.0, 0.0, 0.0, 1.0])   # beam along z
        num_spurions = len(spurions)
        self.num_spurions = num_spurions
        if num_spurions > 0:
            self.register_buffer(
                "spurions_4v_buffer",
                torch.tensor(spurions, dtype=torch.float32),
                persistent=False,
            )

        # ---- Scalar embedding.
        self.embedding_h = nn.Linear(in_s_channels, n_h_hidden)

        # ---- GNN stack. Block 0: (1 + num_spurions) -> n_v_hidden vec
        # channels; later blocks keep n_v_hidden.
        self.gnn_blocks = nn.ModuleList([
            LorentzNetKNNBlock(
                n_h_in=n_h_hidden,
                n_h_out=n_h_hidden,
                n_v_in=(1 + num_spurions) if i == 0 else n_v_hidden,
                n_v_out=n_v_hidden,
                c_weight=c_weight,
                use_phi_m=use_phi_m,
                n_node_attr=in_s_channels if use_node_attr else 0,
            )
            for i in range(n_gnn_layers)
        ])

        # ---- Scalar bridge to LGATrSlim's input scalar dimension.
        bridge_in = n_h_hidden + (in_s_channels if concat_original else 0)
        self.bridge_s = nn.Sequential(
            nn.Linear(bridge_in, hidden_s_channels),
            nn.LayerNorm(hidden_s_channels),
        )

        # ---- L-GATr-slim input vector channel count.
        in_v_channels = n_v_hidden + (1 if concat_original else 0)
        self.in_v_channels = in_v_channels

        # ---- Learnable CLS scalar (vector part is zero -- a learnable
        # 4-vector has a learnable direction and would break equivariance).
        # Named cls_token so the tagging optimizer excludes it from weight decay.
        if global_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_s_channels))
            nn.init.normal_(self.cls_token, std=0.02)

        # ---- L-GATr-slim.
        self.net = LGATrSlim(
            in_v_channels=in_v_channels,
            out_v_channels=1,                     # vector sink (unused for readout)
            hidden_v_channels=hidden_v_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=num_classes,           # classification logits
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_ratio=attn_ratio,
            num_layers_mlp=num_layers_mlp,
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
        }

    def forward(self, x, v, mask, points=None):
        """
        Parameters
        ----------
        x : (B, F, P)    scalar features
        v : (B, 4, P)    4-vectors in (px, py, pz, E)  [weaver convention]
        mask : (B, 1, P) bool/float
        points : (B, D, P) or None
            Coordinates used to build the kNN graph when knn_metric='deltaR'
            (typically D=2, (eta, phi)). Ignored when knn_metric='minkowski'.

        Returns
        -------
        (B, num_classes) logits
        """
        s_in = x.transpose(1, 2)                                     # (B, P, F)
        mask_b = mask.transpose(1, 2).bool().squeeze(-1)            # (B, P)
        B, P, _ = s_in.shape

        # ---- Static kNN graph (computed ONCE, reused across blocks).
        # `knn` expects (px, py, pz, E) for minkowski (the weaver input order),
        # so it is called on v directly, before the internal reorder. Padded
        # particles are excluded from the neighbour search via mask_b.
        if self.knn_metric == "minkowski":
            idx = knn(v, self.knn_k, metric="minkowski", mask=mask_b)
        else:  # 'deltaR'
            if points is None:
                raise ValueError(
                    "knn_metric='deltaR' requires `points` (e.g., (eta, phi))"
                )
            idx = knn(points, self.knn_k, metric="deltaR", mask=mask_b)

        # validity mask for the gathered neighbours (handles the n_real < k case)
        nbr_valid = gather_neighbors(mask_b.unsqueeze(-1), idx).squeeze(-1)
        nbr_mask = nbr_valid & mask_b.unsqueeze(-1)                  # (B, P, K)
        # Exclude self: sparse jets (n_real < knn_k) let topk fill the neighbour list from the
        # tied -inf/self slots, and the realness masks above would re-admit the node's own index
        # as a spurious self message. The knn distance already masks self to -inf.
        nbr_mask = nbr_mask & (idx != torch.arange(idx.shape[1], device=idx.device)[None, :, None])

        # ---- Internal reorder to (E, px, py, pz).
        v_in = v.transpose(1, 2)                                     # (B, P, 4) (px,py,pz,E)
        v_etxyz = v_in[..., [3, 0, 1, 2]]                            # (B, P, 4) (E,px,py,pz)

        # ---- Scalar embedding.
        h = self.embedding_h(s_in)                                   # (B, P, n_h)

        # ---- Initial vector stream: particle 4-vec + spurion 4-vecs.
        particle_v = v_etxyz.unsqueeze(2)                            # (B, P, 1, 4)
        if self.num_spurions > 0:
            spurions = self.spurions_4v_buffer.to(particle_v.dtype)  # (S, 4)
            spurion_broadcast = spurions[None, None, :, :].expand(B, P, -1, -1)
            x_vec = torch.cat([particle_v, spurion_broadcast], dim=2)
        else:
            x_vec = particle_v
        # x_vec: (B, P, 1 + num_spurions, 4)

        # ---- GNN stack (same static graph across all blocks). The raw input scalars
        # ride along as per-layer node attributes (official LorentzNet's node_attr).
        node_attr = s_in if self.use_node_attr else None
        for block in self.gnn_blocks:
            h, x_vec = block(h, x_vec, idx, nbr_mask, node_attr=node_attr)

        # ---- Zero out padded slots before the bridge / L-GATr-slim.
        m_h = mask_b.unsqueeze(-1).to(h.dtype)
        h = h * m_h
        x_vec = x_vec * m_h.unsqueeze(-1)

        # ---- Scalar bridge.
        if self.concat_original:
            s = self.bridge_s(torch.cat([h, s_in], dim=-1))
        else:
            s = self.bridge_s(h)

        # ---- Vector stream into L-GATr-slim: optional raw-particle skip.
        if self.concat_original:
            v_lgatr = torch.cat([x_vec, particle_v], dim=2)
        else:
            v_lgatr = x_vec
        # v_lgatr: (B, P, in_v_channels, 4)  in (E, px, py, pz)

        mask_seq = mask_b

        # ---- Learnable CLS token (scalar learnable, vector zero).
        is_global = None
        if self.global_token:
            cls_v = torch.zeros(
                B, 1, self.in_v_channels, 4,
                device=v_lgatr.device, dtype=v_lgatr.dtype,
            )
            cls_s = self.cls_token.expand(B, -1, -1)
            v_lgatr = torch.cat([cls_v, v_lgatr], dim=1)
            s = torch.cat([cls_s, s], dim=1)
            mask_seq = torch.cat([
                torch.ones(B, 1, device=mask_seq.device, dtype=torch.bool),
                mask_seq,
            ], dim=1)
            is_global = torch.zeros(s.shape[:2], dtype=torch.bool, device=s.device)
            is_global[:, 0] = True

        # ---- L-GATr-slim. Boolean key-padding mask (True = attend).
        attn_mask = mask_seq[:, None, None, :]                       # (B, 1, 1, seq)
        _, s_out = self.net(v_lgatr, s, attn_mask=attn_mask)
        # s_out: (B, seq, num_classes) -- Lorentz-invariant class logits.

        # ---- Aggregate.
        if self.global_token:
            output = s_out[is_global]
        else:
            valid = mask_seq.to(s_out.dtype)
            output = (s_out * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
        return output
