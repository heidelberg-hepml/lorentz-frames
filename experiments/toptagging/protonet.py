import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tensorframes.nn.edge_conv import EdgeConv
from tensorframes.reps import TensorReps
from tensorframes.nn.mlp import MLPWrapped


class ProtoNet(nn.Module):
    def __init__(
        self,
        num_blocks,
        in_reps,
        hidden_reps,
        out_reps,
        checkpoint_blocks=False,
        **mlp_kwargs,
    ):
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        assert num_blocks >= 2

        # convert x_reps from string to proper TensorReps objects
        in_reps = TensorReps(in_reps)
        hidden_reps = TensorReps(hidden_reps)
        out_reps = TensorReps(out_reps)

        # build edgeconv blocks
        edgeconv_kwargs = {
            "aggr": "add",
            "spatial_dim": 3,
            "hidden_channels": [hidden_reps.dim, hidden_reps.dim],
            "radial_module": None,
            "angular_module": None,
            "concatenate_receiver_features_in_mlp1": False,
            "concatenate_receiver_features_in_mlp2": False,
            "use_edge_feature_product": False,
            **mlp_kwargs,
        }
        first_block = EdgeConv(
            in_reps=in_reps,
            out_channels=hidden_reps.dim,
            **edgeconv_kwargs,
        )
        middle_blocks = []
        for _ in range(num_blocks - 2):
            middle_blocks.append(
                EdgeConv(
                    in_reps=hidden_reps,
                    out_channels=hidden_reps.dim,
                    **edgeconv_kwargs,
                )
            )
        last_block = EdgeConv(
            in_reps=hidden_reps,
            out_channels=out_reps.dim,
            **edgeconv_kwargs,
        )
        self.blocks = nn.ModuleList([first_block, *middle_blocks, last_block])

    def forward(self, x, pos, edge_index, lframes, batch):
        # loop through edge_conv blocks
        for block in self.blocks:
            if self.checkpoint_blocks:
                x = checkpoint(
                    block,
                    x=x,
                    pos=pos,
                    edge_index=edge_index,
                    batch=batch,
                    lframes=lframes,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x=x, pos=pos, edge_index=edge_index, batch=batch, lframes=lframes
                )
        return x
