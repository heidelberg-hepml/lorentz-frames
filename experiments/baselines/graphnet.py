import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


from torch_geometric.nn import MessagePassing
from tensorframes.nnhep.mlp import MLP


class EdgeConv(MessagePassing):
    def __init__(
        self,
        channels,
        num_layers_mlp1,
        num_layers_mlp2,
        aggr="add",
        dropout_prob=None,
    ):
        super().__init__(aggr=aggr)
        self.mlp1 = (
            MLP(
                in_shape=[channels * 2],
                out_shape=[channels],
                hidden_layers=num_layers_mlp1,
                hidden_channels=channels,
                dropout_prob=dropout_prob,
            )
            if num_layers_mlp1 > 0
            else nn.Identity()
        )
        self.mlp2 = (
            MLP(
                in_shape=[channels],
                out_shape=[channels],
                hidden_layers=num_layers_mlp2,
                hidden_channels=channels,
                dropout_prob=dropout_prob,
            )
            if num_layers_mlp2 > 0
            else nn.Identity()
        )

    def forward(self, x, edge_index):
        # batch = (batch, batch)
        x_aggr = self.propagate(
            edge_index,
            x=x,
        )
        x_aggr = self.mlp2(x_aggr)
        return x_aggr

    def message(self, x_i, x_j):
        x = x_j
        x = torch.cat((x, x_i), dim=-1)
        x = self.mlp1(x)
        return x


class GraphNet(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_blocks,
        num_layers_mlp1=2,
        num_layers_mlp2=0,
        aggr="add",
        checkpoint_blocks=False,
        dropout_prob=None
    ):
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks

        self.linear_in = nn.Linear(in_channels, hidden_channels)
        self.linear_out = nn.Linear(hidden_channels, out_channels)
        self.blocks = nn.ModuleList(
            [
                EdgeConv(
                    hidden_channels,
                    num_layers_mlp1,
                    num_layers_mlp2,
                    aggr=aggr,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, edge_index):
        x = self.linear_in(x)
        for block in self.blocks:
            if self.checkpoint_blocks:
                x = checkpoint(
                    block,
                    x=x,
                    edge_index=edge_index,
                )
            else:
                x = block(
                    x=x,
                    edge_index=edge_index,
                )
        x = self.linear_out(x)
        return x
