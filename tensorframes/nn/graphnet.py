import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


from tensorframes.nn.tfmessage_passing import TFMessagePassing
from tensorframes.nn.mlp import MLP
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

    def forward(self, x, lframes, edge_index, batch):
        lframes = (lframes, lframes)

        x_aggr = self.propagate(
            edge_index,
            x=x,
            lframes=lframes,
            batch=batch,
        )
        x_aggr = self.mlp2(x_aggr)
        return x_aggr

    def message(self, x_i, x_j, lframes_i, lframes_j):
        x = x_j
        x = torch.cat((x, x_i), dim=-1)
        x = self.mlp1(x)
        return x


class TFGraphNet(nn.Module):
    """Baseline graphnet.

    Combines num_blocks EdgeConv blocks.

    Parameters
    ----------
    in_reps : str
        Input representation.
    hidden_reps : str
        Representation during message passing.
    out_reps : str
        Output representation.
    num_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    increase_hidden_channels : int
        Factor by which the key, query, and value size is increased over the default value of
        hidden_channels / num_heads.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    """

    def __init__(
        self,
        in_reps: str,
        hidden_reps: str,
        out_reps: str,
        num_blocks: int,
        num_layers_mlp1: int = 2,
        num_layers_mlp2: int = 0,
        aggr="add",
        checkpoint_blocks=False,
        dropout_prob=None,
    ):
        super().__init__()
        in_reps = TensorReps(in_reps)
        hidden_reps = TensorReps(hidden_reps)
        out_reps = TensorReps(out_reps)
        self.checkpoint_blocks = checkpoint_blocks

        self.linear_in = nn.Linear(in_reps.dim, hidden_reps.dim)
        self.linear_out = nn.Linear(hidden_reps.dim, out_reps.dim)
        self.blocks = nn.ModuleList(
            [
                EdgeConv(
                    hidden_reps,
                    num_layers_mlp1,
                    num_layers_mlp2,
                    aggr=aggr,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, inputs, lframes, edge_index, batch):
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor with shape (..., num_items, in_reps.dim)
            Input data
        lframes : LFrames
            Local frames used for message passing
        edge_index : Tensor with shape (2, num_edges)

        Returns
        -------
        outputs : Tensor with shape (..., num_items, out_reps.dim)
            Outputs
        """
        x = self.linear_in(inputs)
        for block in self.blocks:
            if self.checkpoint_blocks:
                x = checkpoint(
                    block,
                    x=x,
                    lframes=lframes,
                    edge_index=edge_index,
                    batch=batch,
                )
            else:
                x = block(
                    x=x,
                    lframes=lframes,
                    edge_index=edge_index,
                    batch=batch,
                )
        outputs = self.linear_out(x)
        return outputs
