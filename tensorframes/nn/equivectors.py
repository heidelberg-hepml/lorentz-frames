import torch
from torch.nn import Identity, Softplus
from torch_geometric.nn import MessagePassing

from tensorframes.nn.mlp import MLP


class EquivariantVectors(MessagePassing):
    """
    Equivariantly predict learned vectors

    Node and edge attributes are supported,
    the most basic setting is to only use edge attributes
    corresponding to the invariant masses m_{ij} of pairs
    """

    def __init__(
        self,
        n_vectors,
        in_nodes,
        in_edges,
        hidden_channels,
        num_layers,
        operation="diff",
        nonlinearity=None,
        dropout_prob=None,
    ):
        super().__init__()
        self.n_vectors = n_vectors
        self.operation = self.get_operation(operation)
        self.nonlinearity = self.get_activation(nonlinearity)
        in_channels = 2 * in_nodes + in_edges

        self.mlp = MLP(
            in_shape=[in_channels],
            out_shape=n_vectors,
            hidden_channels=hidden_channels,
            hidden_layers=num_layers,
            dropout_prob=dropout_prob,
        )

    def forward(self, x, fm, edge_attr, edge_index):
        vecs = self.propagate(edge_index, x=x, fm=fm, edge_attr=edge_attr)
        vecs = vecs.reshape(-1, self.n_vectors, 4)
        assert torch.isfinite(vecs).all()
        return vecs

    def message(self, x_i, x_j, fm_i, fm_j, edge_attr):
        prefactor = torch.cat([x_i, x_j], dim=-1)
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
            return self.get_fmj
        else:
            raise Exception(
                f"Invalid operation {operation}. Options are (add, diff, single)."
            )

    def get_fmj(self, fm_i, fm_j):
        return fm_j

    def get_activation(self, nonlinearity):
        if nonlinearity == None:
            return Identity()
        elif nonlinearity == "exp":
            return torch.exp
        elif nonlinearity == "softplus":
            return Softplus()
        else:
            raise Exception(
                f"Invalid nonlinearity: {nonlinearity}. Options are (None, exp, softplus)."
            )
