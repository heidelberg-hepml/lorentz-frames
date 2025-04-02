from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa
from xformers.ops import AttentionBias, memory_efficient_attention

from tensorframes.lframes.lframes import (
    InverseLFrames,
    LowerIndices,
)
from tensorframes.reps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform

# Masked out attention logits are set to this constant (a finite replacement for -inf):
_MASKED_OUT = float("-inf")


class InvariantParticleAttention(torch.nn.Module):
    def __init__(self, attn_reps):
        super().__init__()
        self.transform = TensorRepsTransform(TensorReps(attn_reps))

    def forward(self, q_local, k_local, v_local, lframes, **attn_kwargs):
        """
        Strategy
        1) Transform q, k, v into global frame
        2) Apply attention in global frame
        3) Transform output back into local frame

        Comments
        - dimensions: *dims (optional), H (head), N (particles), C (channels)
        - TODO: dynamically reshape attn_mask for default torch attention (in attn function?)

        Parameters
        ----------
        q_local: torch.tensor of shape (*dims, H, N, C)
        k_local: torch.tensor of shape (*dims, H, N, C)
        v_local: torch.tensor of shape (*dims, H, N, C)
        lframes: (*dims, N, 4, 4)
        attn_kwargs: dict
            Optional arguments that are passed on to attention
        """
        # check input shapes
        assert q_local.shape == k_local.shape == v_local.shape
        assert q_local.shape[:-3] == lframes.shape[:-3]  # *dims match
        assert q_local.shape[-2] == lframes.shape[-3]  # N matches

        # insert lframes head dimension
        lframes = lframes.reshape(*q_local.shape[:-3], 1, lframes.shape[-3], 4, 4)
        lframes = lframes.expand(*q_local.shape[:-1], 4, 4)

        inv_lframes = InverseLFrames(lframes)
        lower_inv_lframes = LowerIndices(inv_lframes)

        q_global = self.transform(q_local, inv_lframes)
        k_global = self.transform(k_local, lower_inv_lframes)
        v_global = self.transform(v_local, inv_lframes)

        # (B, H, N, C) format required for xformers
        shape = q_global.shape
        q_global = q_global.reshape(-1, *shape[-3:])
        k_global = k_global.reshape(-1, *shape[-3:])
        v_global = v_global.reshape(-1, *shape[-3:])

        # attention (in global frame)
        out_global = scaled_dot_product_attention(
            q_global,
            k_global,
            v_global,
            **attn_kwargs,
        )

        out_global = out_global.view(*shape)  # (*dims, H, N, C)

        # transform out back into local frame
        out_local = self.transform(out_global, lframes)
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
        in_dtype = query.dtype
        query, key, value = (
            query.to(torch.float32),
            key.to(torch.float32),
            value.to(torch.float32),
        )
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
        return out.to(in_dtype)
    return torch_sdpa(query, key, value, attn_mask=attn_mask, is_causal=is_causal)
