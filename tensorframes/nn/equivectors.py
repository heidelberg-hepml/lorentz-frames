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
        operation="single",
        nonlinearity="exp",
        dropout_prob=None,
    ):
        super().__init__()
        self.n_vectors = n_vectors
        self.operation = self.get_operation(operation)
        self.nonlinearity = self.get_nonlinearity(nonlinearity)
        in_channels = 2 * in_nodes + in_edges

        self.mlp = MLP(
            in_shape=[in_channels],
            out_shape=n_vectors,
            hidden_channels=hidden_channels,
            hidden_layers=num_layers,
            dropout_prob=dropout_prob,
        )

    def forward(self, x, fm, edge_attr, edge_index, spurions):
        """
        Args:
            x: scalar features shape: (batch, n_scalar)
            fm: fourmomenta shape: (batch, 4)
            edge_attr: edge attributes shape: (edges, n_edge_attr)
            edge_index: edge indices shape: (edges, 2)
            spurions: spurions for affine symmetry breaking shape: (batch, n_vectors* 4)
        """
        vecs = self.propagate(
            edge_index, x=x, fm=fm, edge_attr=edge_attr, spurions=spurions
        )
        vecs = vecs.reshape(-1, self.n_vectors, 4)
        assert torch.isfinite(vecs).all()
        return vecs

    def message(self, x_i, x_j, fm_i, fm_j, edge_attr, spurions_i):
        prefactor = torch.cat([x_i, x_j], dim=-1)
        prefactor = torch.cat([prefactor, edge_attr], dim=-1)
        prefactor = self.mlp(prefactor)
        prefactor = self.nonlinearity(prefactor)

        fm_rel = self.operation(fm_i, fm_j)

        out = torch.einsum("...j,...k->...jk", prefactor, fm_rel)
        out = out + spurions_i.reshape(out.shape)
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
