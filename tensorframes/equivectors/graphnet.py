import torch
import math
from torch.nn import Identity, Softplus
from torch_geometric.nn import MessagePassing

from tensorframes.nn.mlp import MLP
from tensorframes.equivectors.base import EquiVectors
from tensorframes.utils.lorentz import lorentz_squarednorm
from tensorframes.utils.utils import (
    build_edge_index_fully_connected,
    get_edge_index_from_ptr,
)


class EquiGraphNet(EquiVectors, MessagePassing):
    """
    Node and edge attributes are supported,
    the most basic setting is to only use edge attributes
    corresponding to the invariant masses m_{ij} of pairs
    """

    def __init__(
        self,
        n_vectors,
        in_nodes,
        hidden_channels,
        num_layers_mlp,
        num_blocks=1,
        include_edges=True,
        operation="single",
        nonlinearity="exp",
        dropout_prob=None,
    ):
        super().__init__()
        assert num_blocks == 1, "More to come"

        self.include_edges = include_edges
        self.n_vectors = n_vectors
        self.operation = self.get_operation(operation)
        self.nonlinearity = self.get_nonlinearity(nonlinearity)

        in_edges = 1 if include_edges else 0
        in_channels = 2 * in_nodes + in_edges
        self.mlp = MLP(
            in_shape=[in_channels],
            out_shape=n_vectors,
            hidden_channels=hidden_channels,
            hidden_layers=num_layers_mlp,
            dropout_prob=dropout_prob,
        )

        if include_edges:
            self.register_buffer("edge_inited", torch.tensor(False, dtype=torch.bool))
            self.register_buffer("edge_mean", torch.zeros(0))
            self.register_buffer("edge_std", torch.ones(1))

    def forward(self, fourmomenta, scalars=None, ptr=None):
        # get edge_index and batch from ptr
        in_shape = fourmomenta.shape[:-1]
        if scalars is None:
            scalars = torch.zeros_like(fourmomenta[..., []])
        if len(in_shape) > 1:
            assert ptr is None, "ptr only supported for sparse tensors"
            edge_index, batch = build_edge_index_fully_connected(
                fourmomenta, remove_self_loops=True
            )
            fourmomenta = fourmomenta.reshape(math.prod(in_shape), 4)
            scalars = scalars.reshape(math.prod(in_shape), scalars.shape[-1])
        else:
            if ptr is None:
                # assume batch contains only one particle
                ptr = torch.tensor([0, len(fourmomenta)], device=fourmomenta.device)
            edge_index = get_edge_index_from_ptr(ptr)
            batch = None

        # calculate and standardize edge attributes
        if self.include_edges:
            mij2 = lorentz_squarednorm(
                fourmomenta[edge_index[0]] + fourmomenta[edge_index[1]]
            ).unsqueeze(-1)
            edge_attr = mij2.clamp(min=1e-10).log()
            if not self.edge_inited:
                self.edge_mean = edge_attr.mean()
                self.edge_std = edge_attr.std().clamp(min=1e-5)
                self.edge_inited.fill_(True)
            edge_attr = (edge_attr - self.edge_mean) / self.edge_std
        else:
            edge_attr = None

        # message-passing
        vecs = self.propagate(
            edge_index, s=scalars, fm=fourmomenta, edge_attr=edge_attr, batch=batch
        )
        vecs = vecs.reshape(*in_shape, self.n_vectors, 4)
        assert torch.isfinite(vecs).all()
        return vecs

    def message(self, s_i, s_j, fm_i, fm_j, edge_attr=None):
        prefactor = torch.cat([s_i, s_j], dim=-1)
        if edge_attr is not None:
            prefactor = torch.cat([prefactor, edge_attr], dim=-1)
        prefactor = self.mlp(prefactor)
        prefactor = self.nonlinearity(prefactor)

        fm_rel = self.operation(fm_i, fm_j)

        out = torch.einsum("...j,...k->...jk", prefactor, fm_rel)
        return out.flatten(-2, -1)

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
            return Identity()
        elif nonlinearity == "exp":
            return lambda x: torch.clamp(x, min=-10, max=10).exp()
        elif nonlinearity == "softplus":
            return Softplus()
        else:
            raise ValueError(
                f"Invalid nonlinearity {nonlinearity}. Options are (None, exp, softplus)."
            )
