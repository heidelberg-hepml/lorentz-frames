from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa
from xformers.ops import AttentionBias, memory_efficient_attention

from tensorframes.lframes.lframes import (
    LFrames,
    InverseLFrames,
    LowerIndices,
)
from tensorframes.reps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.utils.utils import to_nd

# Masked out attention logits are set to this constant (a finite replacement for -inf):
_MASKED_OUT = float("-inf")


class InvariantParticleAttention(torch.nn.Module):
    def __init__(self, attn_reps):
        super().__init__()
        self.transform = TensorRepsTransform(TensorReps(attn_reps))

    def forward(
        self, q_local, k_local, v_local, lframes, attn_mask=None, is_causal=False
    ):
        """
        dimensions: H (head), N (particles), C (channels)
        q_local, k_local, v_local: (H, N, C)
        """
        # hacky solution for amplitude transformer (clean this up later)
        in_shape = q_local.shape
        if len(in_shape) == 4 and len(lframes.matrices.shape) == 4:
            mat = lframes.matrices.reshape(-1, 4, 4)
            lframes = LFrames(mat)
            q_local, k_local, v_local = [
                x.permute(0, 2, 1, 3)
                .reshape(-1, x.shape[-3], x.shape[-1])
                .permute(1, 0, 2)
                for x in [q_local, k_local, v_local]
            ]

        # prepare lframes trafos
        # have to add head dimension
        assert len(q_local.shape) == 3 and len(lframes.matrices.shape) == 3
        matrices = lframes.matrices.unsqueeze(0).repeat(
            q_local.shape[0], *(1,) * len(lframes.shape), 1, 1
        )
        matrices = to_nd(matrices, 3)
        lframes = LFrames(matrices)
        inv_lframes = InverseLFrames(lframes)
        lower_inv_lframes = LowerIndices(inv_lframes)

        # transform q, k, v into global frame
        shape_in = q_local.shape
        q_local, k_local, v_local = (
            to_nd(q_local, 2),
            to_nd(k_local, 2),
            to_nd(v_local, 2),
        )  # (H*N, C)
        q_global = self.transform(q_local, inv_lframes)
        k_global = self.transform(k_local, lower_inv_lframes)
        v_global = self.transform(v_local, inv_lframes)
        q_global, k_global, v_global = (
            q_global.view(*shape_in),
            k_global.view(*shape_in),
            v_global.view(*shape_in),
        )  # (H, N, C)

        # attention (in global frame)
        q_global, k_global, v_global = (
            to_nd(q_global, 4),
            to_nd(k_global, 4),
            to_nd(v_global, 4),
        )  # (1, H, N, C) format required for xformers
        out_global = scaled_dot_product_attention(
            q_global, k_global, v_global, attn_mask=attn_mask, is_causal=is_causal
        )
        out_global = out_global.view(*shape_in)  # (H, N, C)

        # transform out back into local frame
        out_global = to_nd(out_global, 2)
        out_local = self.transform(out_global, lframes)
        out_local = out_local.view(*shape_in)

        # more tricks
        if in_shape != out_local.shape:
            out_local = (
                out_local.permute(1, 0, 2)
                .reshape(in_shape[0], in_shape[2], in_shape[1], in_shape[3])
                .permute(0, 2, 1, 3)
            )
        assert out_local.shape == in_shape
        return out_local


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Union[AttentionBias, Tensor]] = None,
    is_causal=False,
) -> Tensor:
    """Execute (vanilla) scaled dot-product attention.

    Dynamically dispatch to xFormers if attn_mask is an instance of xformers.ops.AttentionBias

    Parameters
    ----------
    query : Tensor
        of shape [batch, head, item, d]
    key : Tensor
        of shape [batch, head, item, d]
    value : Tensor
        of shape [batch, head, item, d]
    attn_mask : Optional[Union[AttentionBias, Tensor]]
        Attention mask
    is_causal: bool

    Returns
    -------
    Tensor
        of shape [batch, head, item, d]
    """
    if isinstance(attn_mask, AttentionBias):
        assert (
            not is_causal
        ), "is_causal=True not implemented yet for xformers attention"
        if key.shape[1] != query.shape[1]:  # required to make multi_query work
            key = key.expand(key.shape[0], query.shape[1], *key.shape[2:])
            value = value.expand(value.shape[0], query.shape[1], *value.shape[2:])
        query = query.transpose(
            1, 2
        )  # [batch, head, item, d] -> [batch, item, head, d]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        out = memory_efficient_attention(
            query.contiguous(),
            key.contiguous(),
            value,
            attn_bias=attn_mask,
        )
        out = out.transpose(1, 2)  # [batch, item, head, d] -> [batch, head, item, d]
        return out
    return torch_sdpa(query, key, value, attn_mask=attn_mask, is_causal=is_causal)
