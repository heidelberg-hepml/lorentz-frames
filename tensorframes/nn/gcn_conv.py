from typing import Union

import torch
from torch_geometric.utils import add_remaining_self_loops, degree

from tensorframes.lframes.lframes import LFrames
from tensorframes.nn.tfmessage_passing import TFMessagePassing
from tensorframes.reps.irreps import Irreps
from tensorframes.reps.tensorreps import TensorReps


class GCNConv(TFMessagePassing):
    """GCNConv class represents a Graph Convolutional Network layer in the tensorframes
    formalism."""

    def __init__(
        self, in_reps: Union[TensorReps, Irreps], out_reps: Union[TensorReps, Irreps]
    ) -> None:
        """Initialize the GCNConv layer.

        Args:
            in_reps (Union[TensorReps, Irreps]): The input representations.
            out_reps (Union[TensorReps, Irreps]): The output representations.
        """
        super().__init__(
            params_dict={
                "x": {"type": "local", "rep": in_reps},
                "norm": {"type": None, "rep": None},
            }
        )
        self.linear = torch.nn.Linear(in_reps.dim, out_reps.dim)

    def forward(
        self, edge_index: torch.Tensor, x: torch.Tensor, lframes: LFrames
    ) -> torch.Tensor:
        """Performs the forward pass of the GCNConv layer.

        Args:
            edge_index (torch.Tensor): The edge indices of the graph.
            x (torch.Tensor): The input node features.
            lframes (LFrames): The LFrames object of the given nodes.

        Returns:
            torch.Tensor: The output node features after the forward pass.
        """

        # add remaining self_loops to edge_index
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.propagate(edge_index, x=x, lframes=lframes, norm=norm)

        return x

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """Computes the message passed between nodes in a graph convolutional network.

        Args:
            x_j (torch.Tensor): The input tensor representing the features of neighboring nodes.
            norm (torch.Tensor): The normalization tensor, calculated from the degree of the nodes.

        Returns:
            torch.Tensor: The computed message tensor.
        """
        return self.linear(x_j) * norm.view(-1, 1)


if __name__ == "__main__":
    in_reps = Irreps("1x0n+1x1n")

    layer = GCNConv(in_reps, in_reps)

    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 4], [1, 0, 2, 1, 3, 2, 2]], dtype=torch.long
    )

    x = torch.randn(5, 4)

    l_frames_tensor = torch.eye(3).unsqueeze(0).repeat(5, 1, 1)

    lframes = LFrames(l_frames_tensor)

    out = layer(edge_index, x, lframes)

    print("out: ", out)
