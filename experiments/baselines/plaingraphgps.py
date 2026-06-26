"""Plain GraphGPS hybrid: interleaved static-MPNN + torch-MHA (GraphGPS recipe).

GraphGPS (Rampasek et al., 2022, arXiv:2205.12454) differs from this repo's other
graph-transformer hybrids in *how* the local and global stages are combined: each
GPS layer runs a local message-passing block and a global attention block **in
parallel on the same input** and sums them, rather than stacking a GNN stage and
then a transformer stage (cf. PlainGraphTrans, which is sequential). The precise
per-layer recipe (paper App. D, Eq. 9-11) is

    X_M = Norm( Dropout( MPNN(X) ) + X )        # local branch, own residual+norm
    X_T = Norm( Dropout( Attn(X) ) + X )        # global branch, own residual+norm
    X'  = Norm( Dropout( FFN(X_M + X_T) ) + (X_M + X_T) )   # fuse by SUM, then FFN

with the two branch residuals being exactly what makes summing two sub-networks
and stacking many layers trainable. The FFN inner width is 2*dim (paper figure).

This is the "plain", non-equivariant variant:
  * a STATIC kNN graph (built once per forward, reused by every layer), with a
    selectable metric -- 'deltaR' (L2 on (eta, phi)) or 'minkowski' (|interval|
    on the four-momenta) -- and the usual robustness tweaks (k capped at P-1,
    padded nodes excluded), shared with PlainGraphTrans via ``knn``;
  * a plain static message-passing local branch with the default update function
    (message MLP on [h_i, h_j(, e_ij)] -> masked mean -> update MLP), NOT GatedGCN;
  * a plain global branch of torch ``nn.MultiheadAttention`` (NOT L-GATr).

Three GraphGPS ingredients are exposed as toggles, all OFF in the default configs
so that PlainGraphGPS vs PlainGraphTrans isolates the *fusion* (interleaved vs
sequential) and nothing else:
  * ``use_edge_attr`` -- feed the Minkowski edge invariant log|(p_i + p_j)^2| into
    the local MPNN messages. GraphGPS's local MPNN "encodes real edge features",
    but PlainGraphTrans's MPNN does not, so this is off by default for a fair
    head-to-head; turning it on is the "physically-motivated relative PE" ablation;
  * ``use_rwse`` -- Random-Walk Structural Encoding (return probabilities of
    length-1..``rwse_k`` walks on the static kNN graph) concatenated to the node
    inputs. A local SE that stays a Lorentz invariant when the graph is invariant
    (e.g. minkowski kNN). (In a *dynamic*-graph model that rebuilds the kNN graph
    each layer it would have to be recomputed per layer.)
  * ``use_lappe`` -- LapPE: the lowest ``lappe_k`` non-trivial eigenvectors of the
    static-kNN normalized Laplacian, sign-flipped in training (GraphGPS's global PE,
    the one it found most useful on molecules). The *least* motivated encoding for
    jets -- it manufactures absolute node position that constituents already carry as
    (eta, phi), and it tracks the *invented* graph rather than the physics; it is also
    O(B*P^3) per forward (no preprocessing cache). Provided for faithful ablation.

  Jet constituents are never anonymous nodes, so PE/SE is not expected to help here;
  both ``use_rwse`` and ``use_lappe`` are OFF by default and provided purely for ablation.
  * ``norm`` -- 'batch' (default) or 'layer'. GraphGPS uses BatchNorm in every
    one of its 59 dataset configs (the ``gt.batch_norm`` flag; ``gt.layer_norm``
    is never True), so 'batch' is the faithful default; here it is applied over
    the real nodes only, which matches GraphGPS's sparse BatchNorm1d (padded
    slots excluded). 'layer' is the padding-safe per-token alternative for
    ablation (it is unaffected by jet size and batch composition).

Being non-equivariant, it is made Lorentz-equivariant by LLoCa tensorial
message-passing in PlainGraphGPSWrapper (which inherits TaggerWrapper), exactly like
ParticleNetParTGraphGPS: the inputs are canonicalized and the per-particle frames are
passed into the backbone, which transports the local-MPNN neighbours
(``change_local_frame``, typed by ``edge_reps``) and the attention q/k/v
(``LLoCaAttention``, typed by ``attn_reps``). LLoCa is added *purely additively*: for
identity/global frames every transport is skipped and the backbone is bit-identical to
the plain, non-equivariant hybrid (the transport adds no parameters and no init
randomness). The mean-pool readout over the invariant local features needs no jet frame.

Input convention (channels-first, matching ParT/ParticleNet/PlainGraphTrans):
    points:   (N, 2, P)   kNN coordinates (eta, phi), used when knn_metric='deltaR'
    features: (N, C, P)   scalar per-particle features
    v:        (N, 4, P)   four-momenta as (px, py, pz, energy), used for 'minkowski'
    mask:     (N, 1, P)   1 for real particles, 0 for padding
"""

import torch
import torch.nn as nn
from lloca.backbone.attention import LLoCaAttention
from lloca.backbone.particlenet import change_local_frame
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform

from experiments.baselines.particlenettransformer import lloca_transport_attention
from experiments.baselines.plaingraphtrans import gather_neighbors, knn

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU}


def minkowski_edge_attr(v, idx, eps=1e-8):
    """Per-edge Lorentz invariant log|(p_i + p_j)^2| on the static kNN graph.

    v: (B, 4, P) as (px, py, pz, E); idx: (B, P, K). Returns (B, 1, P, K). This is
    the physically-motivated, frame-invariant relative encoding GraphGPS's local
    MPNN can consume in place of PE/SE (toggle ``use_edge_attr``).

    Not a GraphGPS construct -- it is a spiritual sibling of ParT's pairwise
    interaction features (specifically the pair invariant-mass term of the U_ij
    attention bias), routed through the MPNN edge channel instead of the attention
    logits; it stands in for the (molecular) edge features GraphGPS's local MPNN
    consumes, which a jet has none of.
    """
    nbr = gather_neighbors(v, idx)            # (B, 4, P, K)
    psum = v.unsqueeze(-1) + nbr              # (B, 4, P, K), center broadcast over K
    px, py, pz, E = psum[:, 0], psum[:, 1], psum[:, 2], psum[:, 3]
    m2 = E * E - px * px - py * py - pz * pz   # (B, P, K)
    return m2.abs().clamp(min=eps).log().unsqueeze(1)   # (B, 1, P, K)


def rwse_encoding(idx, mask_p, k):
    """Random-Walk Structural Encoding on the static kNN graph.

    For each node, the landing (return) probabilities diag(M^s) for s = 1..k of a
    random walk on the symmetrized, degree-normalized kNN graph M = D^-1 A. It is a
    cheap graph-structural node feature, invariant whenever the graph is invariant
    (e.g. minkowski kNN). idx: (B, P, K); mask_p: (B, P). Returns (B, P, k), with
    padded nodes zeroed.
    """
    B, P, _ = idx.shape
    m = mask_p.to(idx.device, torch.float32)                  # (B, P)
    A = torch.zeros(B, P, P, device=idx.device)
    A.scatter_(2, idx, 1.0)                                    # i -> its kNN neighbours
    A = torch.maximum(A, A.transpose(1, 2))                   # symmetrize
    A = A * m.unsqueeze(1) * m.unsqueeze(2)                   # drop padded rows/cols
    A.diagonal(dim1=1, dim2=2).zero_()                        # no self-loops
    M = A / A.sum(-1, keepdim=True).clamp(min=1.0)            # row-normalized RW matrix
    out, Mp = [], M
    for _ in range(k):
        out.append(torch.diagonal(Mp, dim1=1, dim2=2))       # (B, P): return prob this step
        Mp = torch.bmm(Mp, M)
    return torch.stack(out, dim=-1) * m.unsqueeze(-1)         # (B, P, k)


def lappe_encoding(idx, mask_p, k, training=False):
    """Laplacian-eigenvector positional encoding (LapPE) on the static kNN graph.

    The k lowest non-trivial eigenvectors of the symmetric-normalized Laplacian
    L = I - D^-1/2 A D^-1/2, computed per jet, with GraphGPS's random sign-flip in
    training (LapPE eigenvectors are sign-ambiguous; SignNet is the heavier
    sign-invariant alternative, not implemented here). idx: (B, P, K); mask_p: (B, P).
    Returns (B, P, k) with padded nodes zeroed.

    Padded nodes are decoupled with a large diagonal (3 > the normalized-Laplacian
    spectrum max of 2) so they sort ABOVE the real spectrum -- their eigenvectors are
    then zero on real nodes, and a jet with < k+1 real nodes just gets zero-padded
    LapPE channels.

    Largely UNMOTIVATED for jets (see the module docstring): LapPE manufactures absolute
    node position, which jet constituents already carry as (eta, phi), and it is defined
    on the *invented* kNN graph, so it tracks the graph you built rather than the physics.
    It is also O(B*P^3) per forward (no cross-batch caching, unlike GraphGPS's
    preprocessed PE). Provided as a faithful-GraphGPS ablation toggle only.
    """
    B, P, _ = idx.shape
    m = mask_p.to(torch.float32)                              # (B, P)
    A = torch.zeros(B, P, P, device=idx.device)
    A.scatter_(2, idx, 1.0)
    A = torch.maximum(A, A.transpose(1, 2)) * m.unsqueeze(1) * m.unsqueeze(2)
    A.diagonal(dim1=1, dim2=2).zero_()
    dinv = A.sum(-1).clamp(min=1.0).rsqrt()                   # (B, P): D^-1/2
    eye = torch.eye(P, device=idx.device).expand(B, -1, -1)
    L = eye - dinv.unsqueeze(2) * A * dinv.unsqueeze(1)       # sym-normalized Laplacian
    L = L * m.unsqueeze(1) * m.unsqueeze(2)                   # zero padded rows/cols
    L = L + torch.diag_embed((1.0 - m) * 3.0)                # padded diag -> 3 (decoupled)
    evecs = torch.linalg.eigh(L)[1]                           # ascending eigenvalues
    pe = evecs[..., 1:k + 1]                                  # drop trivial, take k lowest
    if pe.shape[-1] < k:                                      # tiny event -> zero-pad channels
        pe = torch.cat([pe, pe.new_zeros(B, P, k - pe.shape[-1])], dim=-1)
    if training:                                             # GraphGPS sign-flip augmentation
        pe = pe * (torch.randint(0, 2, (B, 1, k), device=idx.device) * 2 - 1).to(pe.dtype)
    return pe * m.unsqueeze(-1)                               # (B, P, k)


class MaskedNorm(nn.Module):
    """LayerNorm (per-token, padding-safe) or masked BatchNorm over real nodes only."""

    def __init__(self, dim, kind="layer"):
        super().__init__()
        if kind not in ("layer", "batch"):
            raise ValueError(f"norm must be 'layer' or 'batch', got '{kind}'")
        self.kind = kind
        self.norm = nn.LayerNorm(dim) if kind == "layer" else nn.BatchNorm1d(dim)

    def forward(self, h, mask_bool):          # h: (B, P, C); mask_bool: (B, P)
        if self.kind == "layer":
            return self.norm(h)
        out = h.clone()
        out[mask_bool] = self.norm(h[mask_bool])   # BatchNorm over the real nodes only
        return out


class GPSLocalMPNN(nn.Module):
    """Local branch (Eq. 7): a plain static-graph message-passing block.

    message m_ij = MLP([h_i, h_j(, e_ij)]); aggregate = masked mean over the kNN
    neighbours; update h' = MLP([h_i, agg]). It carries NO internal residual or
    norm -- the GPS layer owns the external dropout -> residual -> norm (Eq. 9).
    """

    def __init__(self, dim, edge_dim=0, act="relu", in_reps=None):
        super().__init__()
        # in_reps (optional) types the dim-wide hidden features for the LLoCa neighbour
        # transport (e.g. "64x0n+16x1n"); its dim must equal dim. None -> no transport (the
        # transform is built only when in_reps is given, used only when forward gets frames).
        self.trafo = None
        if in_reps is not None:
            reps = TensorReps(in_reps)
            assert reps.dim == dim, f"in_reps.dim {reps.dim} != dim {dim}"
            self.trafo = TensorRepsTransform(reps)
        Act = _ACT[act]
        self.message = nn.Sequential(
            nn.Conv2d(2 * dim + edge_dim, dim, 1), Act(),
            nn.Conv2d(dim, dim, 1), Act(),
        )
        self.update = nn.Sequential(
            nn.Conv1d(2 * dim, dim, 1), Act(),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, h, idx, nbr_mask, edge_attr=None, frames=None):
        # h: (B, C, P); idx: (B, P, K); nbr_mask: (B, P, K) bool; edge_attr: (B, E, P, K)
        B, C, P = h.shape
        K = idx.shape[-1]
        nbr = gather_neighbors(h, idx)                       # (B, C, P, K)
        # LLoCa: express neighbour j (in its own local frame) in centre i's frame before the
        # message. No-op when frames is None (identity path) or in_reps is scalar-only.
        if frames is not None and self.trafo is not None:
            idx_base = torch.arange(B, device=h.device).view(-1, 1, 1) * P
            idx_flat = (idx + idx_base).reshape(-1)
            nbr = change_local_frame(nbr, idx_flat, frames, self.trafo)
        center = h.unsqueeze(-1).expand(-1, -1, -1, K)       # (B, C, P, K)
        msg_in = [center, nbr] + ([edge_attr] if edge_attr is not None else [])
        m = self.message(torch.cat(msg_in, dim=1))           # (B, C, P, K)

        nm = nbr_mask.unsqueeze(1).to(m.dtype)               # (B, 1, P, K)
        m = m * nm
        count = nm.sum(dim=-1).clamp(min=1.0)                # (B, 1, P)
        agg = m.sum(dim=-1) / count                          # (B, C, P): masked mean
        return self.update(torch.cat([h, agg], dim=1))       # (B, C, P)


class GPSLayer(nn.Module):
    """One GraphGPS layer: parallel local-MPNN + global-attention, fused by sum.

    Faithful to the paper's residual/norm placement (Eq. 9-11); see the module
    docstring for the ``norm`` choice.
    """

    def __init__(self, dim, num_heads, edge_dim=0, ffn_ratio=2,
                 dropout=0.0, attn_dropout=0.0, act="relu", norm="batch", in_reps=None):
        super().__init__()
        Act = _ACT[act]
        # in_reps types the local-MPNN hidden features for the LLoCa neighbour transport
        # (additive; no-op for identity frames). None -> non-tensorial original MPNN.
        self.local = GPSLocalMPNN(dim, edge_dim, act, in_reps=in_reps)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)

        self.norm_local = MaskedNorm(dim, norm)
        self.norm_attn = MaskedNorm(dim, norm)
        self.norm_ffn = MaskedNorm(dim, norm)
        self.drop_local = nn.Dropout(dropout)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_ratio * dim), Act(),
            nn.Dropout(dropout), nn.Linear(ffn_ratio * dim, dim),
        )

    def forward(self, h, idx, nbr_mask, key_padding_mask, node_mask, edge_attr=None,
                frames_flat=None, lloca_attn=None):
        # h: (B, P, C); node_mask: (B, P, 1) float; frames_flat: (B*P, 4, 4) for the local
        # transport; lloca_attn: shared LLoCaAttention (None on the identity/global path ->
        # plain attention, bit-identical to the original).
        mask_bool = ~key_padding_mask                                # (B, P)
        # ---- local branch (Eq. 9) ----
        h_cf = (h * node_mask).transpose(1, 2)                       # (B, C, P)
        m_local = self.local(h_cf, idx, nbr_mask, edge_attr, frames=frames_flat).transpose(1, 2)  # (B, P, C)
        h_local = self.norm_local(self.drop_local(m_local) + h, mask_bool)
        # ---- global branch (Eq. 10): LLoCa q/k/v transport for learned frames ----
        if lloca_attn is None:
            a = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        else:
            a = lloca_transport_attention(
                h, self.attn, lloca_attn, key_padding_mask=key_padding_mask, attn_mask=None,
                dropout_p=self.attn.dropout if self.training else 0.0,
            )
        h_attn = self.norm_attn(self.drop_attn(a) + h, mask_bool)
        # ---- fuse by SUM, then FFN (Eq. 11) ----
        h = h_local + h_attn
        h = self.norm_ffn(self.drop_ffn(self.ffn(h)) + h, mask_bool)
        return h * node_mask


class PlainGraphGPS(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # static kNN graph
                 knn_k=16,
                 knn_metric="deltaR",
                 use_edge_attr=False,
                 use_fts_bn=True,
                 # positional/structural encoding (ablation; off by default)
                 use_rwse=False,
                 rwse_k=16,
                 use_lappe=False,
                 lappe_k=8,
                 # GPS layers
                 dim=128,
                 num_layers=10,
                 num_heads=8,
                 # LLoCa tensorial message-passing (additive; a no-op for identity/global frames).
                 # attn_reps types the per-head q/k/v transport (attn_reps.dim * num_heads == dim);
                 # edge_reps types the local-MPNN hidden features (edge_reps.dim == dim). Shared by
                 # every layer (h keeps width dim). None -> non-tensorial (original GraphGPS hybrid).
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
        self.use_edge_attr = use_edge_attr
        self.use_rwse = use_rwse
        self.rwse_k = rwse_k
        self.use_lappe = use_lappe
        self.lappe_k = lappe_k
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
        enc_in = input_dim + (rwse_k if use_rwse else 0) + (lappe_k if use_lappe else 0)
        self.node_encoder = nn.Linear(enc_in, dim)

        edge_dim = 1 if use_edge_attr else 0
        self.layers = nn.ModuleList([
            GPSLayer(dim, num_heads, edge_dim, ffn_ratio, dropout, attn_dropout, act, norm,
                     in_reps=edge_reps)
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
        features = features * mask
        mask_p = mask.squeeze(1)                     # (B, P)

        # LLoCa transport is additive: engaged only for non-trivial frames. The readout is a
        # mean-pool over the (invariant) local features, so no jet frame is needed.
        do_transport = frames is not None and not frames.is_global
        frames_flat = frames.reshape(-1, 4, 4) if do_transport else None
        if do_transport:
            self.lloca_attn.prepare_frames(frames)
        block_lloca = self.lloca_attn if do_transport else None

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # static kNN graph (built once, reused by every GPS layer)
            if self.knn_metric == "minkowski" and v is not None:
                idx = knn(v, self.knn_k, metric="minkowski", mask=mask_p)
            else:
                idx = knn(points, self.knn_k, metric="deltaR", mask=mask_p)
            nbr_mask = gather_neighbors(mask.float(), idx).squeeze(1) > 0.5   # (B, P, K)

            edge_attr = None
            if self.use_edge_attr and v is not None:
                edge_attr = minkowski_edge_attr(v, idx)                       # (B, 1, P, K)
                edge_attr = edge_attr * nbr_mask.unsqueeze(1).to(edge_attr.dtype)

            fts = self.bn_fts(features) * mask if self.bn_fts is not None else features
            h_in = fts.transpose(1, 2)                                        # (B, P, input_dim)
            if self.use_rwse:
                rwse = rwse_encoding(idx, mask_p, self.rwse_k).to(h_in.dtype)  # (B, P, rwse_k)
                h_in = torch.cat([h_in, rwse], dim=-1)
            if self.use_lappe:
                lappe = lappe_encoding(
                    idx, mask_p, self.lappe_k, training=self.training
                ).to(h_in.dtype)                                              # (B, P, lappe_k)
                h_in = torch.cat([h_in, lappe], dim=-1)
            h = self.node_encoder(h_in)                                       # (B, P, dim)

            node_mask = mask_p.unsqueeze(-1).to(h.dtype)                      # (B, P, 1)
            key_padding_mask = ~mask_p                                        # (B, P), True = ignore
            for layer in self.layers:
                h = layer(h, idx, nbr_mask, key_padding_mask, node_mask, edge_attr,
                          frames_flat, block_lloca)

            # masked mean pooling over real particles
            pooled = (h * node_mask).sum(dim=1) / node_mask.sum(dim=1).clamp(min=1.0)
            output = self.head(pooled)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            return output
