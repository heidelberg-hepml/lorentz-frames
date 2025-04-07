import torch
from torch import nn
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from tensorframes.nn.mlp import MLP
from tensorframes.equivectors.base import EquiVectors
from tensorframes.utils.lorentz import lorentz_squarednorm
from tensorframes.utils.utils import (
    build_edge_index_fully_connected,
    get_edge_index_from_ptr,
    get_edge_attr,
)


class EquiEdgeConv(MessagePassing):
    def __init__(
        self,
        in_vectors,
        out_vectors,
        num_scalars,
        hidden_channels,
        num_layers_mlp,
        include_edges=True,
        operation="single",
        nonlinearity="exp",
        fm_norm=False,
        dropout_prob=None,
        aggr="sum",
        layer_norm=False,
    ):
        super().__init__(aggr=aggr)
        assert num_scalars > 0 or include_edges
        self.include_edges = include_edges
        self.layer_norm = layer_norm
        self.operation = self.get_operation(operation)
        self.nonlinearity = self.get_nonlinearity(nonlinearity)
        self.fm_norm = fm_norm

        in_edges = in_vectors if include_edges else 0
        in_channels = 2 * num_scalars + in_edges
        self.mlp = MLP(
            in_shape=[in_channels],
            out_shape=out_vectors,
            hidden_channels=hidden_channels,
            hidden_layers=num_layers_mlp,
            dropout_prob=dropout_prob,
        )

        if include_edges:
            self.register_buffer("edge_inited", torch.tensor(False, dtype=torch.bool))
            self.register_buffer("edge_mean", torch.zeros(0))
            self.register_buffer("edge_std", torch.ones(1))

    def forward(self, fourmomenta, scalars, edge_index, batch=None):
        # calculate and standardize edge attributes
        fourmomenta = fourmomenta.reshape(scalars.shape[0], -1, 4)
        if self.include_edges:
            edge_attr = get_edge_attr(fourmomenta, edge_index)
            if not self.edge_inited:
                self.edge_mean = edge_attr.mean().detach()
                self.edge_std = edge_attr.std().clamp(min=1e-5).detach()
                self.edge_inited.fill_(True)
            edge_attr = (edge_attr - self.edge_mean) / self.edge_std
            edge_attr = edge_attr.reshape(edge_attr.shape[0], -1)

            # related to fourmomenta_float64 option
            edge_attr = edge_attr.to(scalars.dtype)
        else:
            edge_attr = None

        # message-passing
        fourmomenta = fourmomenta.reshape(fourmomenta.shape[0], -1)
        vecs = self.propagate(
            edge_index, s=scalars, fm=fourmomenta, edge_attr=edge_attr, batch=batch
        )

        # equivariant layer normalization
        if self.layer_norm:
            norm = lorentz_squarednorm(vecs.reshape(fourmomenta.shape[0], -1, 4))
            norm = norm.sum(dim=-1).unsqueeze(-1)
            vecs = vecs / norm.clamp(min=1e-5).sqrt()
        return vecs

    def message(self, edge_index, s_i, s_j, fm_i, fm_j, edge_attr=None):
        fm_rel = self.operation(fm_i, fm_j)
        # should not be used with operation "single"
        if self.fm_norm:
            fm_rel_norm = lorentz_squarednorm(fm_rel).unsqueeze(-1)
            fm_rel_norm = fm_rel_norm.abs().sqrt().clamp(min=1e-6)
        else:
            fm_rel_norm = 1.0

        prefactor = torch.cat([s_i, s_j], dim=-1)
        if edge_attr is not None:
            prefactor = torch.cat([prefactor, edge_attr], dim=-1)
        prefactor = self.mlp(prefactor)
        prefactor = self.nonlinearity(prefactor, index=edge_index[0])

        fm_rel = (fm_rel / fm_rel_norm)[:, None, :4]
        prefactor = prefactor.unsqueeze(-1)

        out = prefactor * fm_rel
        out = out.reshape(out.shape[0], -1)
        return out

    def get_operation(self, operation):
        if operation == "diff":
            return torch.sub
        elif operation == "add":
            return torch.add
        elif operation == "single":
            return lambda fm_i, fm_j: fm_j
        else:
            raise ValueError(
                f"Invalid operation {operation}. Options are (add, diff, single)."
            )

    def get_nonlinearity(self, nonlinearity):
        if nonlinearity == None:
            return lambda x, index: x
        elif nonlinearity == "exp":
            return lambda x, index: torch.clamp(x, min=-10, max=10).exp()
        elif nonlinearity == "softplus":
            return lambda x, index: torch.nn.functional.softplus(x)
        elif nonlinearity == "softmax":
            return lambda x, index: softmax(x, index)
        else:
            raise ValueError(
                f"Invalid nonlinearity {nonlinearity}. Options are (None, exp, softplus, softmax)."
            )


class EquiGraphNet(EquiVectors):
    def __init__(
        self,
        n_vectors,
        num_blocks,
        *args,
        hidden_vectors=1,
        **kwargs,
    ):
        super().__init__()

        assert num_blocks >= 1
        in_vectors = [1] + [hidden_vectors] * (num_blocks - 1)
        out_vectors = [hidden_vectors] * (num_blocks - 1) + [n_vectors]
        self.blocks = nn.ModuleList(
            [
                EquiEdgeConv(
                    in_vectors=in_vectors[i],
                    out_vectors=out_vectors[i],
                    *args,
                    **kwargs,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, fourmomenta, scalars=None, ptr=None):
        # get edge_index and batch from ptr
        in_shape = fourmomenta.shape[:-1]
        if scalars is None:
            scalars = torch.zeros_like(fourmomenta[..., []])
        if len(in_shape) > 1:
            assert ptr is None, "ptr only supported for sparse tensors"
            edge_index, batch = build_edge_index_fully_connected(fourmomenta)
            fourmomenta = fourmomenta.reshape(math.prod(in_shape), 4)
            scalars = scalars.reshape(math.prod(in_shape), scalars.shape[-1])
        else:
            if ptr is None:
                # assume batch contains only one particle
                ptr = torch.tensor([0, len(fourmomenta)], device=fourmomenta.device)
            edge_index = get_edge_index_from_ptr(ptr)
            batch = None

        for block in self.blocks:
            fourmomenta = block(
                fourmomenta, scalars=scalars, edge_index=edge_index, batch=batch
            )
        fourmomenta = fourmomenta.reshape(*in_shape, -1, 4)
        return fourmomenta
