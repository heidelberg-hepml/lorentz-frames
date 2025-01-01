import torch
from torch_geometric.nn import MessagePassing

from tensorframes.nn.mlp import MLPWrapped


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
        **mlp_kwargs,
    ):
        super().__init__()
        self.n_vectors = n_vectors
        hidden_channels.append(n_vectors)
        in_channels = 2 * in_nodes + in_edges

        self.mlp = MLPWrapped(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            **mlp_kwargs,
        )

    def forward(self, x, fm, edge_attr, edge_index, batch=None):
        vecs = self.propagate(edge_index, x=x, fm=fm, edge_attr=edge_attr, batch=batch)
        vecs = vecs.reshape(-1, self.n_vectors, 4)
        return vecs

    def message(self, x_i, x_j, fm_i, fm_j, edge_attr, batch_j):
        prefactor = torch.cat([x_i, x_j], dim=-1)
        prefactor = torch.cat([prefactor, edge_attr], dim=-1)
        prefactor = self.mlp(x=prefactor, batch=batch_j)

        fm_rel = fm_i - fm_j
        out = torch.einsum("...j,...k->...jk", prefactor, fm_rel)
        return out.flatten(-2, -1)
