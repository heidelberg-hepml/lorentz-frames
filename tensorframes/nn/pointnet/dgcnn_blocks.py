from typing import Union

import torch
from torch_geometric.nn import knn

from tensorframes.lframes import LFrames
from tensorframes.lframes.learning_lframes import WrappedLearnedLFrames
from tensorframes.lframes.updating_lframes import QuaternionsUpdateLFrames
from tensorframes.nn.edge_conv import EdgeConv
from tensorframes.reps import TensorReps
from tensorframes.utils.point_sampling import CustomPointSampler


class DynamicSAModule(torch.nn.Module):
    """Dynamic Set Abstraction Module (DynamicSAModule) is a module used in the PointNet
    architecture for point cloud processing.

    Attributes:
        k (int): The number of nearest neighbors to consider.
        conv (EdgeConv): The EdgeConv module used for convolutional operations.
        center_sampler (CustomPointSampler): The custom point sampler used for sampling center points.
        out_dim (int): The dimension of the output tensor representations.
        lframes_learner (WrappedLearnedLFrames | None): The module used for learning local frames.
        lframes_updater (QuaternionsUpdateLFrames | None): The module used for updating local frames.
        concat_pos_to_features (bool): Whether to concatenate positions to features.
        concat_lframes_to_features (bool): Whether to concatenate local frames to features.
        knn_feature_reps (TensorReps): The tensor representations used for k-nearest neighbor operations.
        transform_to_global_frame (bool): Whether to transform features to the global frame.
        transform_pos (bool): Whether to transform positions.
        pos_trafo (TransformClass | None): The transformation class for positions.
    """

    def __init__(
        self,
        conv: EdgeConv,
        center_sampler: CustomPointSampler,
        k: int = 20,
        lframes_learner: WrappedLearnedLFrames | None = None,
        lframes_updater: QuaternionsUpdateLFrames | None = None,
        transform_to_global_frame: bool = True,
        concat_pos_to_features: bool = False,
        concat_lframes_to_features: bool = False,
        transform_pos: bool = False,
    ) -> None:
        """Initializes the DynamicSAModule.

        Args:
            conv (EdgeConv): The EdgeConv module used for convolutional operations.
            center_sampler (CustomPointSampler): The custom point sampler used for sampling center points.
            k (int, optional): The number of nearest neighbors to consider. Defaults to 20.
            lframes_learner (WrappedLearnedLocalFramesModule | None, optional): The module used for learning local frames. Defaults to None.
            lframes_updater (QuaternionsUpdateLFrames | None, optional): The module used for updating local frames. Defaults to None.
            transform_to_global_frame (bool, optional): Whether to transform features to the global frame. Defaults to True.
            concat_pos_to_features (bool, optional): Whether to concatenate positions to features. Defaults to False.
            concat_lframes_to_features (bool, optional): Whether to concatenate local frames to features. Defaults to False.
            transform_pos (bool, optional): Whether to transform positions. Defaults to False.
        """
        super().__init__()
        self.k = k
        self.conv = conv
        self.center_sampler = center_sampler
        self.out_dim = conv.in_reps.dim
        self.lframes_learner = lframes_learner
        self.lframes_updater = lframes_updater
        self.concat_pos_to_features = concat_pos_to_features
        self.concat_lframes_to_features = concat_lframes_to_features
        self.knn_feature_reps = conv.in_reps
        self.transform_to_global_frame = transform_to_global_frame
        self.transform_pos = transform_pos
        self.pos_trafo = None
        if self.transform_pos:
            self.pos_trafo = TensorReps("1x1").get_transform_class()

    def forward(
        self,
        x: Union[torch.Tensor, None],
        pos: torch.Tensor,
        batch: Union[torch.Tensor, None],
        lframes: LFrames,
        epoch: Union[int, None] = None,
    ):
        """Performs the forward pass of the DynamicSAModule.

        Args:
            x (torch.Tensor | None): The input tensor.
            pos (torch.Tensor): The positions of the points.
            batch (torch.Tensor | None): The batch indices of the points.
            lframes (LFrames): The local frames of the points.
            epoch (int | None, optional): The current epoch. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
            torch.Tensor: The positions of the selected points.
            torch.Tensor: The batch indices of the selected points.
            LFrames: The updated local frames.
        """
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.int64, device=pos.device)
        idx = self.center_sampler(pos, batch)

        knn_features = torch.tensor([], device=pos.device) if x is None else x
        if self.transform_to_global_frame and x is not None:
            knn_features = self.knn_feature_reps.transform_coeffs(
                knn_features, lframes.view(-1, 3, 3).transpose(-1, -2)
            )
        if self.concat_pos_to_features:
            if self.transform_pos:
                transformed_pos = self.pos_trafo(pos, lframes.view(-1, 3, 3))
                knn_features = torch.cat([knn_features, transformed_pos], dim=-1)
            else:
                knn_features = torch.cat([knn_features, pos], dim=-1)

        if self.concat_lframes_to_features:
            knn_features = torch.cat([knn_features, lframes.view(-1, 9)], dim=-1)
        edge_index = knn(knn_features, knn_features[idx], self.k, batch, batch[idx]).flip([0])

        x_dst = None if x is None else x[idx]
        if self.lframes_learner is not None:
            x_dst, lframes_dst = self.lframes_learner(
                x=(x, x_dst),
                pos=(pos, pos[idx]),
                batch=(batch, batch[idx]),
                edge_index=edge_index,
            )
        else:
            lframes_dst = lframes.index_select(idx)

        if self.lframes_updater is not None:
            x_dst, lframes_dst = self.lframes_updater(
                lframes=lframes_dst,
                x=x_dst,
                batch=batch[idx],
            )

        x = self.conv(
            x=(x, x_dst),
            pos=(pos, pos[idx]),
            lframes=(lframes, lframes_dst),
            batch=(batch, batch[idx]),
            edge_index=edge_index,
        )
        return x, pos[idx], batch[idx], lframes_dst
