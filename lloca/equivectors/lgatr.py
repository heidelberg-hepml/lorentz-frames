import torch
from lgatr import embed_vector, extract_scalar
from lgatr.primitives.attention import scaled_dot_product_attention

from .base import EquiVectors
from ..utils.utils import get_batch_from_ptr
from lloca.nn.attention import get_xformers_attention_mask


class LGATrVectors(EquiVectors):
    def __init__(
        self,
        n_vectors,
        num_scalars,
        net,
    ):
        super().__init__()
        self.n_vectors = n_vectors
        self.net = net(in_s_channels=num_scalars)

    def forward(self, fourmomenta, scalars=None, ptr=None):
        attn_kwargs = {}
        if ptr is not None:
            batch = get_batch_from_ptr(ptr)
            on_cpu = fourmomenta.device == torch.device("cpu")
            mask = get_xformers_attention_mask(
                batch, materialize=on_cpu, dtype=scalars.dtype
            )
            attn_kwargs["attn_mask" if on_cpu else "attn_bias"] = mask
            fourmomenta = fourmomenta.unsqueeze(0)
            scalars = scalars.unsqueeze(0)

        # get query and key from LGATr
        mv = embed_vector(fourmomenta).unsqueeze(-2).to(scalars.dtype)
        out_mv, _ = self.net(mv, scalars, **attn_kwargs)
        out_s = extract_scalar(out_mv).squeeze(-1)
        q, k = torch.chunk(out_s.to(fourmomenta.dtype), chunks=2, dim=-1)

        # prepare value
        v = fourmomenta.repeat(1, 1, self.n_vectors)

        # attention and final reshape
        out = scaled_dot_product_attention(q, k, v, **attn_kwargs)
        out = out.reshape(*out.shape[:-1], self.n_vectors, 4)
        if ptr is not None:
            out = out.squeeze(0)
        return out
