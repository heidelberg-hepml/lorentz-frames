import torch
from lgatr import embed_vector
from lgatr.primitives.attention import scaled_dot_product_attention

from .base import EquiVectors
from ..utils.utils import get_batch_from_ptr
from lloca.nn.attention import get_xformers_attention_mask


class LGATrVectors(EquiVectors):
    def __init__(
        self,
        n_vectors,
        num_scalars,
        hidden_s_channels,
        net,
    ):
        super().__init__()
        self.n_vectors = n_vectors
        self.net = net(
            in_s_channels=num_scalars,
            out_s_channels=2 * n_vectors * hidden_s_channels,
        )

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
        _, out_s = self.net(mv, scalars, **attn_kwargs)
        q, k = torch.chunk(out_s.to(fourmomenta.dtype), chunks=2, dim=-1)
        qs = torch.chunk(q, chunks=self.n_vectors, dim=-1)
        ks = torch.chunk(k, chunks=self.n_vectors, dim=-1)

        # attention and final reshape
        v = fourmomenta
        out = []
        for q, k in zip(qs, ks):
            outi = scaled_dot_product_attention(q, k, v, **attn_kwargs)
            out.append(outi)
        out = torch.stack(out, dim=-2)
        if ptr is not None:
            out = out.squeeze(0)
        return out
