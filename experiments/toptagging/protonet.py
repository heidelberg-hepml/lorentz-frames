import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tensorframes.nn.edge_conv import EdgeConv
from tensorframes.reps import TensorReps


class ProtoNet(nn.Module):
    """ProtoNet: Equivariant network, uses tensorframes EdgeConv, radial and angular embedding with weighth sharing and the possibility of second networks in the EdgeConv layers"""

    def __init__(
        self,
        in_reps,
        hidden_reps,
        out_reps,
        radial_module,
        angular_module,
        checkpoint_blocks=False,
        second_hidden_reps=None,
        aggr="add",
        concatenate_receiver_features_in_mlp1=True,
        concatenate_receiver_features_in_mlp2=True,
        use_edge_feature_product=False,
        hidden_layer_number=None,
        **mlp_kwargs,
    ):
        """Args:
        in_reps (string): string for input dimention of network e.g. "1x0n+1x1n",
        hidden_reps (list[string]): strings for intermediate hidden layers in network, each with 2 linear layers, e.g. ["32x0n+32x1n", "64x0n+64x1n"],
        out_reps (string): string for output dimention of the network, e.g. "1x0n",
        radial_module (tensorframes.nn.embedding.radial.RadialEmbedding) radial embedding for the edge vectors,
        angular_module (tensorframes.nn.embedding.angular.AngularEmbedding) angular/axial embedding for the edge vectors,
        checkpoint_blocks (bool) whether to create checkpoint blocks, Defaults to False,
        second_hidden_reps (list[string]): string for the dimentions of secondary layers in the EdgeConv layers, should have the same dimention as hidden_reps+1. Defaults to None
        hidden_layer_number (list[int]): number of layers in edgeconv blocks
        """
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        num_blocks = len(hidden_reps)
        assert num_blocks >= 2

        if hidden_layer_number is None:
            hidden_layer_number = [2 for i in hidden_reps]
        elif isinstance(hidden_layer_number, int):
            temp = hidden_layer_number
            hidden_layer_number = [temp for i in hidden_reps]

        # convert x_reps from string to proper TensorReps objects
        in_reps = TensorReps(in_reps)
        hidden_reps = [TensorReps(hr) for hr in hidden_reps]
        if second_hidden_reps is None:
            hidden_channels = [
                [hr.dim] * hidden_layer_number[i] for i, hr in enumerate(hidden_reps)
            ]
            second_hidden_channels = [None for hr in hidden_reps]
        else:  # this accounts for the last hidden -> output layer being transfered to the second network
            hidden_channels = [
                [hr.dim] * hidden_layer_number[i]
                for i, hr in enumerate(hidden_reps)
            ]
            second_hidden_reps = [TensorReps(shr) for shr in second_hidden_reps]
            second_hidden_channels = [[shr.dim] for shr in second_hidden_reps]
            assert len(hidden_channels) == len(
                second_hidden_channels
            ), "either none or all of the EdgeConv layers need their own second layer channels"
        out_reps = TensorReps(out_reps)

        self.output_dim = out_reps.dim

        edgeconv_kwargs = {
            "aggr": aggr,
            "concatenate_receiver_features_in_mlp1": concatenate_receiver_features_in_mlp1,
            "concatenate_receiver_features_in_mlp2": concatenate_receiver_features_in_mlp2,
            "use_edge_feature_product": use_edge_feature_product,
            "radial_module": radial_module,
            "angular_module": angular_module,
        }

        # build edgeConv blocks
        first_block = EdgeConv(
            in_reps=in_reps,
            hidden_channels=hidden_channels[0],
            out_channels=hidden_reps[0].dim,
            second_hidden_channels=second_hidden_channels[0],
            **edgeconv_kwargs,
        )

        middle_blocks = []
        for layerID in range(num_blocks - 2):
            middle_blocks.append(
                EdgeConv(
                    in_reps=hidden_reps[layerID],
                    hidden_channels=hidden_channels[layerID + 1],
                    out_channels=hidden_reps[layerID + 1].dim,
                    second_hidden_channels=second_hidden_channels[layerID + 1],
                    **edgeconv_kwargs,
                )
            )
        last_block = EdgeConv(
            in_reps=hidden_reps[-2],
            hidden_channels=hidden_channels[-1],
            out_channels=out_reps.dim,
            second_hidden_channels=second_hidden_channels[-1],
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
