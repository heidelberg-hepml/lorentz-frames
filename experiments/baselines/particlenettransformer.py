"""ParticleNet-Transformer hybrid.

A hybrid tagger: a ParticleNet (EdgeConv) graph backbone feeds a
ParticleTransformer-style encoder with pairwise Lorentz interaction features and a
class token. Adapted from the weaver implementation (the weaver-specific wrapper,
``get_model`` and ``get_loss`` removed).

LLoCa is added *purely additively*: when the wrapper
(``experiments.tagging.wrappers.ParticleNetParTGraphTransWrapper``) passes non-trivial
per-particle ``frames``, the backbone does genuine tensorial message-passing -- the
EdgeConv transports neighbours into the centre frame (lloca change_local_frame, typed by
``hidden_reps_list``) and the attention transports q/k/v between frames (a parameter-free
``LLoCaAttention`` reusing each block's own ``nn.MultiheadAttention`` weights, typed by
``attn_reps``), with the class token riding in the covariant jet frame. For identity/global
frames the transport is skipped entirely and the backbone is bit-identical to the plain,
non-equivariant ParticleNet-ParT (the ``LLoCaAttention`` and the reps add no parameters and
no init randomness, so the identity path is unchanged).

Input convention (channels-first, matches ParT/ParticleNet, NOT the
``(E, px, py, pz)`` convention used elsewhere in this repo):
    points:   (N, 2, P)            kNN coordinates, e.g. (eta, phi)
    features: (N, C, P)            scalar per-particle features
    v:        (N, 4, P)            four-momenta as (px, py, pz, energy)
    mask:     (N, 1, P)            1 for real particles, 0 for padding
"""

import math
import random
import warnings
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from lloca.backbone.attention import LLoCaAttention
from lloca.backbone.particlenet import change_local_frame
from lloca.backbone.particletransformer import _canonical_mask
from lloca.framesnet.frames import Frames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.lorentz import lorentz_eye

_logger = logging.getLogger(__name__)


def lloca_transport_attention(x, mha, lloca_attn, key_padding_mask=None, attn_mask=None, dropout_p=0.0):
    """LLoCa tensorial attention reusing an ``nn.MultiheadAttention``'s own weights.

    Projects q/k/v with ``mha``'s in_proj, hands them to a (parameter-free, already
    ``prepare_frames``-d) ``LLoCaAttention`` -- which transports them between the per-token
    local frames and contracts with the Minkowski metric -- then applies ``mha``'s out_proj.
    This is *only* taken for non-trivial frames; the identity/global path calls ``mha``
    directly, so it stays bit-identical to the plain attention. ``x`` is batch-first
    ``(B, seq, embed)``; ``attn_mask`` is ``(B, num_heads, seq, seq)``.
    """
    bsz, seqlen, embed = x.shape
    num_heads = mha.num_heads
    head_dim = embed // num_heads
    key_padding_mask = _canonical_mask(key_padding_mask, "key_padding_mask", None, "", x.dtype, False)
    attn_mask = _canonical_mask(attn_mask, "attn_mask", None, "", x.dtype, False)
    if key_padding_mask is not None:
        kpm = key_padding_mask.view(bsz, 1, 1, seqlen).expand(-1, num_heads, -1, -1)
        attn_mask = kpm if attn_mask is None else attn_mask + kpm
    q, k, v = F._in_projection_packed(x, x, x, mha.in_proj_weight, mha.in_proj_bias)
    q = q.reshape(bsz, seqlen, num_heads, head_dim).transpose(1, 2).contiguous()
    k = k.reshape(bsz, seqlen, num_heads, head_dim).transpose(1, 2).contiguous()
    v = v.reshape(bsz, seqlen, num_heads, head_dim).transpose(1, 2).contiguous()
    out = lloca_attn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
    out = out.transpose(1, 2).reshape(bsz, seqlen, embed)
    return mha.out_proj(out)


def knn(x, k, metric='deltaR', mask=None):
    # cap k: a node has at most (num_points - 1) possible neighbours, so this
    # avoids a topk out-of-range crash on small/sparse events (e.g. few-particle
    # jets) where k would otherwise exceed the available points.
    num_points = x.size(-1)
    k = min(k, max(1, num_points - 1))
    if metric == 'minkowski':
        # x: (N, 4, P) in (px, py, pz, E). Neighbours are ranked by the absolute
        # Minkowski interval |(x_i - x_j)^2| with metric diag(-1, -1, -1, +1).
        sig = x.new_tensor([-1.0, -1.0, -1.0, 1.0]).view(1, -1, 1)
        xp = x * sig
        msq = (x * xp).sum(dim=1, keepdim=True)  # (N, 1, P): E^2 - p^2
        gram = torch.matmul(x.transpose(2, 1), xp)  # (N, P, P): <x_i, x_j>_mink
        d2 = msq.transpose(2, 1) + msq - 2 * gram  # (N, P, P): (x_i - x_j)^2_mink
        pairwise_distance = -d2.abs()
        # drop self-loops explicitly (lightlike pairs can also reach |interval|=0)
        eye = torch.eye(num_points, dtype=torch.bool, device=x.device).unsqueeze(0)
        pairwise_distance = pairwise_distance.masked_fill(eye, float('-inf'))
        if mask is not None:
            # never rank a padded particle as a neighbour
            pairwise_distance = pairwise_distance.masked_fill(~mask.bool().unsqueeze(1), float('-inf'))
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    else:
        # 'deltaR': squared-L2 on the supplied coordinates. A 2-channel input is the
        # (eta, phi) points (first layer), whose azimuth is PERIODIC: the true
        # particle-particle distance is the shorter arc, so wrap d_phi into [0, pi] before
        # the L2 (a pair straddling the +/-pi seam is adjacent, not ~2*pi apart). The dynamic
        # deeper layers pass higher-dim features (ordinary Euclidean) -> plain gram L2.
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
            idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        else:
            idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx, frames=None, trafo=None):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(-1, num_dims)  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    # LLoCa: express neighbour j (in its own local frame) in centre i's frame before the
    # edge feature fts - x. No-op when frames is None (identity path) -> original behaviour;
    # also skipped when trafo is None (a pure-scalar layer, whose transport is the identity).
    if frames is not None and trafo is not None:
        fts = change_local_frame(fts, idx, frames, trafo)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx, frames=None, trafo=None):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)
    if frames is not None and trafo is not None:
        fts = change_local_frame(fts, idx, frames, trafo)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False,
                 in_reps=None):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1
        # in_reps (optional) types the input features for the LLoCa neighbour transport
        # (e.g. "32x0n+8x1n"); its dim must equal in_feat. None -> no transport (the
        # transform is built but only used when frames are passed to forward).
        self.trafo = None
        if in_reps is not None:
            reps = TensorReps(in_reps)
            assert reps.dim == in_feat, f"in_reps.dim {reps.dim} != in_feat {in_feat}"
            self.trafo = TensorRepsTransform(reps)

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features, knn_metric='deltaR', mask=None, frames=None):

        topk_indices = knn(points, self.k, metric=knn_metric, mask=mask)
        k = topk_indices.size(-1)  # may be capped below self.k for tiny events
        # frames -> LLoCa neighbour transport (no-op / original when frames is None)
        x = self.get_graph_feature(features, k, topk_indices, frames, self.trafo)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)



@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    # clamp the log argument as in lloca's to_ptrapphim: avoids a -inf for a massless
    # particle pointing exactly along -z. Inactive (hence bit-identical to the original)
    # for any physical momentum; only the measure-zero edge case differs.
    rapidity = 0.5 * torch.log((1 + (2 * pz) / (energy - pz).clamp(min=1e-20)).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert (len(outputs) == num_outputs)
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat((
        torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0),
        torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0),
        idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
        idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
    ), dim=0)
    return torch.sparse_coo_tensor(
        i, uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device).to_dense()[:, :, :seq_len, :seq_len]


class PairEmbed(nn.Module):
    def __init__(
            self, pairwise_lv_dim, pairwise_input_dim, dims,
            remove_self_pair=False, use_pre_activation_pair=True, mode='sum',
            normalize_input=True, activation='gelu', eps=1e-8,
            for_onnx=False):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx)
        self.out_dim = dims[-1]

        if self.mode == 'concat':
            input_dim = pairwise_lv_dim + pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend([
                    nn.Conv1d(input_dim, dim, 1),
                    nn.BatchNorm1d(dim),
                    nn.GELU() if activation == 'gelu' else nn.ReLU(),
                ])
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)
        elif self.mode == 'sum':
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)

            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')

    def forward(self, x, uu=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        assert (x is not None or uu is not None)
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0,
                                          device=(x if x is not None else uu).device)
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                    xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                if uu is not None:
                    # (batch, dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == 'concat':
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.mode == 'concat':
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == 'sum':
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)

        # Create padded tensor with zeros for CLS position
        y_padded = torch.zeros(batch_size, self.out_dim, seq_len + 1, seq_len + 1,
                          dtype=y.dtype, device=y.device)
        # Fill in the particle-particle interactions (indices 1:, 1:)
        y_padded[:, :, 1:, 1:] = y
        # CLS-to-CLS, CLS-to-particle, particle-to-CLS remain zero (no physical info)

        return y_padded


class Block(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='gelu',
                 scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def forward(self, x, padding_mask=None, attn_mask=None, lloca_attn=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.
            attn_mask (Tensor, optional): pairwise bias of shape `(batch, num_heads, seq, seq)`.
            lloca_attn (LLoCaAttention, optional): when given (non-trivial frames), q/k/v are
                projected by this block's own MHA weights, transported between frames by
                lloca_attn, then out-projected. When None (identity/global frames) the block
                calls nn.MultiheadAttention directly -> bit-identical to the plain backbone.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """


        residual = x
        x = self.pre_attn_norm(x)
        if lloca_attn is None:
            # identity / global frames: ordinary multi-head attention (the original path)
            am = attn_mask
            if am is not None and am.dim() == 4:
                am = am.reshape(-1, am.size(-2), am.size(-1))  # (batch*num_heads, seq, seq)
            x = self.attn(x, x, x, key_padding_mask=padding_mask,
                          attn_mask=am)[0]  # (seq_len, batch, embed_dim)
        else:
            # learned frames: tensorial transport, reusing this block's MHA projection weights
            xb = lloca_transport_attention(
                x.transpose(0, 1), self.attn, lloca_attn,
                key_padding_mask=padding_mask, attn_mask=attn_mask,
                dropout_p=self.attn.dropout if self.training else 0.0,
            )
            x = xb.transpose(0, 1)  # back to (seq_len, batch, embed_dim)

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class ParticleNetParTGraphTrans(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 use_input_concat=True, #change to false if ablation implies so
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],

                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128],  # transformer width (fallback when attn_reps is None)
                 # LLoCa tensorial message-passing (purely additive; a no-op for identity/global
                 # frames). attn_reps types the per-head q/k/v for the attention transport, so
                 # embed_dim = attn_reps.dim * num_heads. hidden_reps_list[i] types EdgeConv i's
                 # input for the neighbour transport (None entry -> that layer is not transported).
                 # Both default to typed reps; the transport itself is only taken for learned frames.
                 attn_reps="8x0n+2x1n",
                 hidden_reps_list=None,
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 block_params=None,
                 use_fusion=True,
                 use_fts_bn=True,
                 fc_params=[],
                 activation='gelu',
                 bias=True,  # toggle the pairwise additive attention bias (ParT-style U_ij)
                 knn_metric='deltaR',  # first-layer kNN: 'deltaR' (eta-phi L2) or 'minkowski' (4-momenta)
                 #
                 #trim=True,
                 for_inference=False,
                 use_amp=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.use_input_concat = use_input_concat

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dim)

        if hidden_reps_list is None:
            hidden_reps_list = [None] * len(conv_params)
        assert len(hidden_reps_list) == len(conv_params)
        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dim if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels,
                                                 cpu_mode=for_inference, in_reps=hidden_reps_list[idx]))


        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())
        else:
            out_chn = conv_params[-1][-1][-1] #if no fusion, output dimension of gnn is the last edgeconv block's

        #self.trimmer = SequenceTrimmer(enabled=trim and not for_inference) | No Trimmer
        self.for_inference = for_inference
        self.use_amp = use_amp

        # transformer width: from attn_reps (per head) when given, else the legacy embed_dims.
        # the parameter-free LLoCaAttention transports q/k/v only for learned frames.
        if attn_reps is not None:
            attn_reps_t = TensorReps(attn_reps)
            embed_dim = attn_reps_t.dim * num_heads
            self.lloca_attn = LLoCaAttention(attn_reps_t, num_heads)
        else:
            embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
            self.lloca_attn = None

        self.knn_metric = knn_metric
        if knn_metric not in ('deltaR', 'minkowski'):
            raise ValueError(f"knn_metric must be 'deltaR' or 'minkowski', got '{knn_metric}'")

        bridge_in_dim = out_chn + input_dim if use_input_concat else out_chn
        self.bridge = nn.Linear(bridge_in_dim, embed_dim)
        self.bridge_norm = nn.LayerNorm(embed_dim)

        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info('cfg_block: %s' % str(cfg_block))


        self.pair_extra_dim = pair_extra_dim
        # `bias` toggles the pairwise additive attention bias: when False, no
        # PairEmbed is built and the transformer runs without interaction features.
        #self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity() | No Direct Embedding algo for input (GNN used)
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if bias and pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, points, features, v=None, mask=None, uu=None, uu_idx=None,
                frames=None, cls_frames=None):

        '''
        Points: (N, 2, P)
        Features: (N, C, P)
        Vectors: (N, 4, P) [px,py,pz,energy]
        Mask: (N, 1, P)
        frames: Frames (N, P, 4, 4) per-particle local frames, or None. The LLoCa transport
            is taken only when frames is given AND not global; identity/global frames take
            the plain path (bit-identical to the non-LLoCa backbone).
        cls_frames: Frames (N, 4, 4) covariant jet frame for the prepended CLS token's slot
            (used only on the transport path); None -> identity slot.
        '''

        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)
        else:
            mask=mask.bool()

        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9

        # LLoCa transport is purely additive: only engaged for non-trivial frames.
        do_transport = frames is not None and not frames.is_global
        frames_flat = frames.reshape(-1, 4, 4) if do_transport else None

        with torch.no_grad(): #extra pairwise feature handling
            if not self.for_inference:
                if uu_idx is not None:
                    # Rebuild dense (N, C', P, P)
                    uu = build_sparse_tensor(uu, uu_idx, features.size(-1))

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            if self.use_fts_bn:
                fts = self.bn_fts(features) * mask
            else:
                fts = features
            mask_knn = mask.squeeze(1)  # (N, P): padded particles excluded from the kNN
            outputs = []
            for idx, conv in enumerate(self.edge_convs):
                # first layer: static graph from the input geometry (eta-phi or, for
                # 'minkowski', the four-momenta); deeper layers: dynamic feature graph
                if idx == 0 and self.knn_metric == 'minkowski' and v is not None:
                    pts, metric = v + coord_shift, 'minkowski'
                else:
                    pts, metric = (points if idx == 0 else fts) + coord_shift, 'deltaR'
                fts = conv(pts, fts, knn_metric=metric, mask=mask_knn, frames=frames_flat) * mask
                if self.use_fusion:
                    outputs.append(fts)
            if self.use_fusion:
                fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask
    #---
            if self.use_input_concat:
                bridge_in = torch.cat([fts, features], dim=1)
            else:
                bridge_in = fts

            #linear projection
            x=bridge_in.permute(0,2,1) # (N, P, in_dim)
            x=self.bridge_norm(self.bridge(x))

            x = x.permute(1, 0, 2).contiguous()  # (P, N, embed_dim)

            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                # pair embed internally pads for CLS -> (N, num_heads, P+1, P+1); the Block
                # reshapes to (N*num_heads, P+1, P+1) for the plain-attention path.
                attn_mask = self.pair_embed(v, uu)

            #prepend CLS token
            cls = self.cls_token.expand(1, x.size(1), -1)
            x= torch.cat([cls, x], dim=0)


            pad = ~mask.squeeze(1)  # (N,P)
            # CLS is always valid
            padding_mask = torch.cat([torch.zeros_like(pad[:, :1], dtype=torch.bool), pad], dim=1)

            # On the transport path, prepare the per-token frames (CLS in the covariant jet
            # frame, then the particles) once for the shared, parameter-free LLoCaAttention.
            block_lloca = None
            if do_transport:
                B = features.size(0)
                if cls_frames is not None:
                    cls_mat = cls_frames.matrices.to(frames.dtype)
                else:
                    cls_mat = lorentz_eye((B,), device=frames.device, dtype=frames.dtype)
                seq_mat = torch.cat([cls_mat.unsqueeze(1), frames.matrices], dim=1)  # (N, P+1, 4, 4)
                self.lloca_attn.prepare_frames(Frames(seq_mat, is_global=False, is_identity=False))
                block_lloca = self.lloca_attn

            for block in self.blocks:
                x = block(x, padding_mask=padding_mask, attn_mask=attn_mask, lloca_attn=block_lloca)

            x_cls = self.norm(x[0]) #is a norm necessary here? it shows up in ParT but ParT's class token is used only in the final 2 blocks to aggregate and so needs to be normalized, but this doesn't appear to be the case in GraphTrans?

            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)

            return output
