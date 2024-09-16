import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tensorframes.nn.edge_conv_easy import EdgeConv
from tensorframes.reps import TensorReps
from tensorframes.nn.mlp import MLPWrapped
from experiments.logger import LOGGER


class ProtoNet(nn.Module):
    def __init__(
        self,
        in_reps,
        hidden_reps,
        out_reps,
        radial_module,
        angular_module,
        checkpoint_blocks=False,
        concatenate_receiver_features_in_mlp1=concatenate_receiver_features_in_mlp1,
        **mlp_kwargs,
    ):
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        num_blocks = len(hidden_reps)
        assert num_blocks >= 2


        # convert x_reps from string to proper TensorReps objects
        in_reps = TensorReps(in_reps) #TensorReps(4)
        hidden_reps = [TensorReps(hr) for hr in hidden_reps] #[TensorReps(32),TensorReps(128),TensorReps(256)]
        hidden_channels = [[hr.dim]*2 for hr in hidden_reps] #[[32,32],[128,128],[256,256]]
        out_reps = TensorReps(out_reps)

        #build edgeConv blocks
        first_block = EdgeConv(
            in_reps=in_reps,
            hidden_channels=hidden_channels[0],
            out_channels=hidden_reps[0].dim,
            aggr="add",
            radial_module=radial_module,
            angular_module=angular_module,
            concatenate_receiver_features_in_mlp1=concatenate_receiver_features_in_mlp1
        )
        
        middle_blocks = []
        for layerID in range(num_blocks - 2):
            middle_blocks.append(
                EdgeConv(
                    in_reps=hidden_reps[layerID],
                    hidden_channels=hidden_channels[layerID+1],
                    out_channels=hidden_reps[layerID+1].dim,
                    aggr="add",
                    radial_module=radial_module,
                    angular_module=angular_module,
                    concatenate_receiver_features_in_mlp1=concatenate_receiver_features_in_mlp1
                )
            )
        last_block = EdgeConv(
            in_reps=hidden_reps[-2],
            hidden_channels=hidden_channels[-1],
            out_channels=out_reps.dim,
            aggr="add",
            radial_module=radial_module,
            angular_module=angular_module,
            concatenate_receiver_features_in_mlp1=concatenate_receiver_features_in_mlp1
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
