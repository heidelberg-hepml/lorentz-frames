import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


from tensorframes.nn.tfmessage_passing import TFMessagePassing
from tensorframes.nnhep.mlp import MLP
from tensorframes.reps.tensorreps import TensorReps


class EdgeConv(TFMessagePassing):
    def __init__(
        self,
        reps,
        num_layers_mlp1,
        num_layers_mlp2,
        aggr="add",
        dropout_prob=None,
    ):
        super().__init__(aggr=aggr, params_dict={"x": {"type": "local", "rep": reps}})
        self.mlp1 = MLP(
            in_shape=[reps.dim * 2],
            out_shape=[reps.dim],
            hidden_layers=num_layers_mlp1,
            hidden_channels=reps.dim,
            dropout_prob=dropout_prob,
        )
        self.mlp2 = (
            MLP(
                in_shape=[reps.dim],
                out_shape=[reps.dim],
                hidden_layers=num_layers_mlp2,
                hidden_channels=reps.dim,
                dropout_prob=dropout_prob,
            )
            if num_layers_mlp2 > 0
            else nn.Identity()
        )

    def forward(self, x, lframes, edge_index):
        lframes = (lframes, lframes)

        x_aggr = self.propagate(
            edge_index,
            x=x,
            lframes=lframes,
        )
        x_aggr = self.mlp2(x_aggr)
        return x_aggr

    def message(self, x_i, x_j, lframes_i, lframes_j):
        x = x_j
        x = torch.cat((x, x_i), dim=-1)
        x = self.mlp1(x)
        return x


class TFGraphNet(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_reps,
        out_channels,
        num_blocks,
        num_layers_mlp1=2,
        num_layers_mlp2=0,
        aggr="add",
        checkpoint_blocks=False,
        **mlp_kwargs,
    ):
        super().__init__()
        hidden_reps = TensorReps(hidden_reps)
        self.checkpoint_blocks = checkpoint_blocks

        self.linear_in = nn.Linear(in_channels, hidden_reps.dim)
        self.linear_out = nn.Linear(hidden_reps.dim, out_channels)
        self.blocks = nn.ModuleList(
            [
                EdgeConv(
                    hidden_reps,
                    num_layers_mlp1,
                    num_layers_mlp2,
                    aggr=aggr,
                    **mlp_kwargs,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, lframes, edge_index):
        x = self.linear_in(x)
        for block in self.blocks:
            if self.checkpoint_blocks:
                x = checkpoint(
                    block,
                    x=x,
                    lframes=lframes,
                    edge_index=edge_index,
                )
            else:
                x = block(
                    x=x,
                    lframes=lframes,
                    edge_index=edge_index,
                )
        x = self.linear_out(x)
        return x
