import torch
from torch import nn
import math

from tensorframes.equivectors.base import EquiVectors
from tensorframes.utils.utils import get_xformers_attention_mask, get_batch_from_ptr
from tensorframes.nn.attention import scaled_dot_product_attention
from tensorframes.utils.lorentz import lorentz_squarednorm


class EquiLinear(nn.Module):
    def __init__(self, in_vectors, out_vectors, num_scalars=None):
        super().__init__()
        if num_scalars is not None:
            self.linear_scalar = nn.Linear(num_scalars, num_scalars)
        self.weight = nn.Parameter(torch.empty(out_vectors, in_vectors))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, vectors, scalars=None):
        if scalars is not None:
            scalars = self.linear_scalar(scalars)

        # linear combination of vectors without mixing their components
        vectors = torch.einsum("ij,...jk->...ik", self.weight, vectors)
        return vectors, scalars


class EquiLinearTimelike(nn.Module):
    def __init__(self, in_vectors, out_vectors):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_vectors, in_vectors))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, vectors):
        # difference to EquiLinear: apply exp to the weights
        vectors = torch.einsum("ij,...jk->...ik", self.weight.exp(), vectors)
        return vectors


class EquiAttention(nn.Module):
    def __init__(
        self, in_vectors, out_vectors, hidden_scalars, increase_attention_vectors
    ):
        super().__init__()
        self.q_linear = EquiLinear(
            in_vectors,
            in_vectors * increase_attention_vectors,
            num_scalars=hidden_scalars,
        )
        self.k_linear = EquiLinear(
            in_vectors,
            in_vectors * increase_attention_vectors,
            num_scalars=hidden_scalars,
        )
        self.v_linear = EquiLinearTimelike(in_vectors, out_vectors)

    def forward(self, vectors, scalars, attn_mask=None):
        # layer normalization
        norm = lorentz_squarednorm(vectors).unsqueeze(-1)
        vectors = vectors / norm.abs().clamp(min=1e-5).sqrt()

        # compute queries and keys
        q_v, q_s = self.q_linear(vectors, scalars)
        k_v, k_s = self.k_linear(vectors, scalars)
        inner_product = torch.tensor(
            [1, -1, -1, -1], device=q_v.device, dtype=q_v.dtype
        )
        inner_product = inner_product.view(*([1] * len(q_v.shape[:-1])), 4)

        def embed(vectors, scalars, apply_factor=False):
            if apply_factor:
                vectors = vectors * inner_product
            vectors = vectors.flatten(-2, -1)
            x = torch.cat([vectors, scalars], -1)
            return x

        q, k = embed(q_v, q_s, apply_factor=False), embed(k_v, k_s, apply_factor=True)

        # compute values (they have to be timelike)
        v_v = self.v_linear(vectors)
        v = v_v.flatten(-2, -1)

        # attention
        out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.reshape(*vectors.shape[:-2], -1, 4)
        return out


class EquiTransformer(EquiVectors):
    def __init__(
        self,
        n_vectors,
        num_scalars,
        num_blocks,
        hidden_scalars=32,
        increase_attention_vectors=8,
        hidden_vectors=1,
    ):
        super().__init__()
        in_vectors = [1] + [hidden_vectors] * (num_blocks - 1)
        out_vectors = [hidden_vectors] * (num_blocks - 1) + [n_vectors]
        self.linear_in = nn.Linear(num_scalars, hidden_scalars)
        self.blocks = nn.ModuleList(
            [
                EquiAttention(
                    in_vectors=in_vectors[i],
                    out_vectors=out_vectors[i],
                    hidden_scalars=hidden_scalars,
                    increase_attention_vectors=increase_attention_vectors,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, fourmomenta, scalars=None, ptr=None):
        assert (lorentz_squarednorm(fourmomenta) > 0).all()
        # construct attn_mask
        if ptr is None:
            attn_mask = None
        else:
            batch = get_batch_from_ptr(ptr)
            attn_mask = get_xformers_attention_mask(
                batch,
                materialize=fourmomenta.device == torch.device("cpu"),
                dtype=fourmomenta.dtype,
            )

        vectors = fourmomenta.unsqueeze(-2)
        scalars = self.linear_in(scalars)
        for block in self.blocks:
            vectors = block(vectors, scalars, attn_mask=attn_mask)
        assert (lorentz_squarednorm(vectors) > 0).all()
        return vectors
