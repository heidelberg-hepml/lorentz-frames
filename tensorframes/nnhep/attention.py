from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa
from xformers.ops import AttentionBias, memory_efficient_attention

from tensorframes.lframes import ChangeOfLFrames, LFrames
from tensorframes.reps import TensorReps
from tensorframes.utils.utils import to_nd

# Masked out attention logits are set to this constant (a finite replacement for -inf):
_MASKED_OUT = float("-inf")


class InvariantParticleAttention(torch.nn.Module):
    def __init__(self, attn_reps):
        super().__init__()
        self.transform = TensorReps(attn_reps).get_transform_class()

    def forward(
        self, q_local, k_local, v_local, lframes, attn_mask=None, is_causal=False
    ):
        # TODO: Clean this up

        # assume that the first dimension is the head dimension
        # -> expand lframes across this dimension
        matrices = lframes.matrices
        matrices = matrices.unsqueeze(0).repeat(
            q_local.shape[0], *(1,) * len(matrices.shape)
        )
        matrices = to_nd(matrices, 3)
        lframes = LFrames(matrices)
        lframes_inv = lframes.inverse_lframes()

        # transformation matrices with lowered indices (multiply with metric)
        lframes_inv_lower_matrices = torch.einsum(
            "...ij,...jk->...ik", lframes.metric, lframes_inv.matrices
        )
        lframes_inv_lower = LFrames(lframes_inv_lower_matrices)

        # collapse batch dimension (required for tensorreps.py)
        bh_shape1 = q_local.shape
        q_local = to_nd(q_local, 2)
        k_local = to_nd(k_local, 2)
        v_local = to_nd(v_local, 2)

        # transform into global frame
        q_global = self.transform(q_local, lframes_inv)
        k_global = self.transform(k_local, lframes_inv_lower)
        v_global = self.transform(v_local, lframes_inv)

        # undo collapse
        q_global = q_global.view(*bh_shape1)
        k_global = k_global.view(*bh_shape1)
        v_global = v_global.view(*bh_shape1)

        # add batch dimension if needed
        bh_shape2 = q_global.shape[:-2]
        q_global = to_nd(q_global, 4)
        k_global = to_nd(k_global, 4)
        v_global = to_nd(v_global, 4)

        out_global = scaled_dot_product_attention(
            q_global, k_global, v_global, attn_mask=attn_mask, is_causal=is_causal
        )

        # Return batch dimensions to inputs
        out_global = out_global.view(*bh_shape2, *out_global.shape[-2:])

        # collapse batch dimension
        out_global = to_nd(out_global, 2)

        # transform into local frame
        out_local = self.transform(out_global, lframes)

        # undo collapse
        out_local = out_local.view(*bh_shape1)
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
