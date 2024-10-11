import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from torch_geometric.nn.conv import EdgeConv
from tensorframes.reps import TensorReps
from tensorframes.nn.mlp import MLPWrapped
from torchvision.ops import MLP


class NonEquiNet(nn.Module):
    """NonEquiNet: Non-Equivariant network, uses torch_geometric EdgeConv"""

    def __init__(
        self,
        in_reps,
        hidden_reps,
        out_reps,
        hidden_channels=None,
        checkpoint_blocks=False,
        **mlp_kwargs,
    ):
        """Args:
        in_reps (string): string for input dimention of network e.g. "1x0n+1x1n",
        hidden_reps (list[string]): strings for intermediate hidden layers in network, each with 2 linear layers, e.g. ["32x0n+32x1n", "64x0n+64x1n"],
        out_reps (string): string for output dimention of the network, e.g. "1x0n",
        checkpoint_blocks (bool) whether to create checkpoint blocks, Defaults to False,
        """
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks

        num_blocks = len(hidden_reps) + 2
        assert num_blocks >= 2

        # convert x_reps from string to proper TensorReps objects
        in_reps = TensorReps(in_reps)
        hidden_reps = TensorReps(hidden_reps)
        out_reps = TensorReps(out_reps)

        if hidden_channels == None:
            self.hc1 = [hidden_reps.dim, hidden_reps.dim, hidden_reps.dim]
            self.hc2 = [hidden_reps.dim, hidden_reps.dim, hidden_reps.dim]
            self.hc3 = [hidden_reps.dim, hidden_reps.dim, out_reps.dim]

            self.mlp1 = MLP(in_channels=in_reps.dim * 2, hidden_channels=self.hc1)
            first_block = EdgeConv(nn=self.mlp1, aggr="add")

            middle_blocks = []
            for _ in range(num_blocks - 2):

                mlp2 = MLP(in_channels=hidden_reps.dim * 2, hidden_channels=self.hc2)
                middle_blocks.append(EdgeConv(nn=mlp2, aggr="add"))

            self.mlp3 = MLP(in_channels=hidden_reps.dim * 2, hidden_channels=self.hc3)
            last_block = EdgeConv(nn=self.mlp3, aggr="add")
        else:
            self.mlp1 = MLP(
                in_channels=in_reps.dim * 2, hidden_channels=[hidden_channels[0]] * 3
            )
            first_block = EdgeConv(nn=self.mlp1, aggr="add")

            middle_blocks = []
            for _ in range(num_blocks - 2):
                mlp2 = MLP(
                    in_channels=hidden_channels[_] * 2,
                    hidden_channels=[hidden_channels[_ + 1]] * 3,
                )
                middle_blocks.append(EdgeConv(nn=mlp2, aggr="add"))

            self.mlp3 = MLP(
                in_channels=hidden_channels[-2] * 2,
                hidden_channels=[*[hidden_channels[-1]] * 2, out_reps.dim],
            )
            last_block = EdgeConv(nn=self.mlp3, aggr="add")

        self.blocks = nn.ModuleList([first_block, *middle_blocks, last_block])

    def forward(self, x, pos, edge_index, batch):
        # loop through edge_conv blocks
        for block in self.blocks:
            if self.checkpoint_blocks:
                x = checkpoint(
                    block,
                    x=x,
                    edge_index=edge_index,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x=x,
                    edge_index=edge_index,
                )
        return x
