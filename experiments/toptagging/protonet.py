import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tensorframes.nn.edge_conv import EdgeConv
from tensorframes.reps import TensorReps
from tensorframes.nn.mlp import MLPWrapped
from experiments.logger import LOGGER


class ProtoNet(nn.Module):
    """
    ProtoNet: Equivariant network, uses tensorframes EdgeConv, radial and angular embedding with weighth sharing and the possibility of second networks in the EdgeConv layers

    Args:
        in_reps (string): string for input dimention of network e.g. "1x0n+1x1n",
        hidden_reps (list[string]): strings for intermediate hidden layers in network, each with 2 linear layers, e.g. ["32x0n+32x1n", "64x0n+64x1n"],
        out_reps (string): string for output dimention of the network, e.g. "1x0n",
        radial_module (tensorframes.nn.embedding.radial.RadialEmbedding) radial embedding for the edge vectors,
        angular_module (tensorframes.nn.embedding.angular.AngularEmbedding) angular/axial embedding for the edge vectors,
        checkpoint_blocks (bool) whether to create checkpoint blocks, Defaults to False,
        second_hidden_reps (list[string]): string for the dimentions of secondary layers in the EdgeConv layers, should have the same dimention as hidden_reps+1. Defaults to None

    """

    def __init__(
        self,
        in_reps,
        hidden_reps,
        out_reps,
        radial_module,
        angular_module,
        checkpoint_blocks=False,
        second_hidden_reps=None,
        **mlp_kwargs,
    ):
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        num_blocks = len(hidden_reps)
        assert num_blocks >= 2

        # convert x_reps from string to proper TensorReps objects
        in_reps = TensorReps(in_reps)
        hidden_reps = [TensorReps(hr) for hr in hidden_reps]
        if second_hidden_reps is None:
            hidden_channels = [[hr.dim] * 2 for hr in hidden_reps]
            second_hidden_channels = [None for hr in hidden_reps]
        else:  # this accounts for the last hidden -> output layer being transfered to the second network
            hidden_channels = [[hr.dim] * 3 for hr in hidden_reps]
            second_hidden_reps = [TensorReps(shr) for shr in second_hidden_reps]
            second_hidden_channels = [[shr.dim] for shr in second_hidden_reps]
            assert (
                len(hidden_channels) == len(second_hidden_channels) - 1
            ), "either none or all of the EdgeConv layers need their own second layer channels"
        out_reps = TensorReps(out_reps)

        self.output_dim = out_reps.dim

        # build edgeConv blocks
        first_block = EdgeConv(
            in_reps=in_reps,
            hidden_channels=hidden_channels[0],
            out_channels=hidden_reps[0].dim,
            aggr="add",
            radial_module=radial_module,
            angular_module=angular_module,
            second_hidden_channels=second_hidden_channels[0],
        )

        middle_blocks = []
        for layerID in range(num_blocks - 2):
            middle_blocks.append(
                EdgeConv(
                    in_reps=hidden_reps[layerID],
                    hidden_channels=hidden_channels[layerID + 1],
                    out_channels=hidden_reps[layerID + 1].dim,
                    aggr="add",
                    radial_module=radial_module,
                    angular_module=angular_module,
                    second_hidden_channels=second_hidden_channels[layerID + 1],
                )
            )
        last_block = EdgeConv(
            in_reps=hidden_reps[-2],
            hidden_channels=hidden_channels[-1],
            out_channels=out_reps.dim,
            aggr="add",
            radial_module=radial_module,
            angular_module=angular_module,
            second_hidden_channels=second_hidden_channels[-1],
        )

        self.blocks = nn.ModuleList([first_block, *middle_blocks, last_block])
        for i, l in enumerate(self.blocks):
            LOGGER.info(f"layer{i}: {l}")

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
