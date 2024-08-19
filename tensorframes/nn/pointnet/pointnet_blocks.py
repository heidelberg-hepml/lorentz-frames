from typing import Union

import torch
from torch_geometric.nn import knn, radius
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter
from torch_scatter import scatter_min

from tensorframes.lframes import ChangeOfLFrames, LFrames
from tensorframes.lframes.learning_lframes import WrappedLearnedLFrames
from tensorframes.lframes.updating_lframes import QuaternionsUpdateLFrames
from tensorframes.nn.edge_conv import EdgeConv
from tensorframes.nn.mlp import MLPWrapped
from tensorframes.reps import Irreps, TensorReps
from tensorframes.utils.point_sampling import CustomPointSampler


class SAModule(torch.nn.Module):
    """Self-Attention Module (SAModule) for PointNet architecture.

    Attributes:
        r (float): Radius for neighbor sampling.
        max_num_neighbors (int): Maximum number of neighbors to consider.
        conv (EdgeConv): EdgeConv module for point-wise feature transformation.
        center_sampler (CustomPointSampler): CustomPointSampler module for center point sampling.
        out_dim (int): Output dimension of the EdgeConv module.
        lframes_updater (QuaternionsUpdateLFrames | None): LFrames updater module.
    """

    def __init__(
        self,
        r: float,
        conv: EdgeConv,
        center_sampler: CustomPointSampler,
        max_num_neighbors: int = 64,
        lframes_learner: WrappedLearnedLFrames | None = None,
        lframes_updater: QuaternionsUpdateLFrames | None = None,
    ) -> None:
        """Initializes a new instance of the SAModule class.

        Args:
            r (float): Radius for neighbor sampling.
            conv (EdgeConv): EdgeConv module for point-wise feature transformation.
            center_sampler (CustomPointSampler): CustomPointSampler module for center point sampling.
            max_num_neighbors (int, optional): Maximum number of neighbors to consider. Defaults to 64.
            lframes_learner (WrappedLearnedLFrames | None, optional): The module used for learning local frames. Defaults to None.
            lframes_updater (QuaternionsUpdateLFrames | None, optional): LFrames updater module. Defaults to None.
        """
        super().__init__()
        self.r = r
        self.max_num_neighbors = max_num_neighbors
        self.conv = conv
        self.center_sampler = center_sampler
        self.out_dim = conv.in_reps.dim
        self.lframes_learner = lframes_learner
        self.lframes_updater = lframes_updater

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        lframes: LFrames,
        epoch: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, LFrames]:
        """Forward pass of the SAModule.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in) representing the features.
            pos (torch.Tensor): Input tensor of shape (N, d) representing the positions.
            batch (torch.Tensor): Input tensor of shape (N,) representing the batch indices.
            lframes (LFrames): Input LFrames object.
            epoch (int | None): The current epoch (optional).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, LFrames]: Tuple containing:
                - x (torch.Tensor): Output tensor of shape (N, C_out) representing the transformed features.
                - pos (torch.Tensor): Output tensor of shape (N, d) representing the transformed positions.
                - batch (torch.Tensor): Output tensor of shape (N,) representing the transformed batch indices.
                - lframes_dst (LFrames): Updated LFrames object. with matrices of shape (N, 3, 3).
        """
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.int64, device=pos.device)
        idx = self.center_sampler(pos, batch)
        # note that if there are more point then max_num_neighbors, they are sampled randomly
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=self.max_num_neighbors
        )
        # print("average number of neighbors: ", len(row) / len(idx), "max_num_neighbors", self.max_num_neighbors)
        edge_index = torch.stack([col, row], dim=0)
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


class GlobalSAModule(torch.nn.Module):
    """Global Set Abstraction Module (GlobalSAModule) class.

    This module performs global set abstraction by selecting a single center point for each batch,
    computing the center of mass (COM) of the remaining points, and choosing the point closest to the COM as the final center.

    Attributes:
        conv (EdgeConv): The EdgeConv module used for convolutional operations.
        lframes_updater (QuaternionsUpdateLFrames | None): The module used for updating local reference frames (LFrames).
        out_dim (int): The dimension of the input tensor representations.
    """

    def __init__(
        self,
        conv: EdgeConv,
        lframes_updater: QuaternionsUpdateLFrames | None = None,
        use_skip_connections: bool = False,
    ) -> None:
        """Initializes a new instance of the GlobalSAModule class.

        Args:
            conv (EdgeConv): The EdgeConv module used for convolutional operations.
            lframes_updater (QuaternionsUpdateLFrames | None, optional): The module used for updating local reference frames (LFrames).
                Defaults to None.
            use_skip_connections (bool, optional): Indicates whether to use skip connections from previous SAM layers (only for DGCNN). Defaults to False.
        """
        super().__init__()

        self.conv = conv
        self.lframes_updater = lframes_updater
        self.out_dim = conv.in_reps.dim
        self.use_skip_connections = use_skip_connections

    def forward(
        self,
        x: torch.Tensor | None,
        pos: torch.Tensor,
        batch: torch.Tensor,
        lframes: LFrames,
        cached_layer_outputs: list[dict] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, LFrames]:
        """Forward pass of the GlobalSAModule.

        Args:
            x (Tensor | None): The input tensor.
            pos (Tensor): The position tensor.
            batch (Tensor): The batch tensor.
            lframes (Tensor): The local reference frames (LFrames) tensor.
            cached_layer_outputs (list[dict] | None): The cached layer outputs (only for DGCNN).

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The output tensor, the selected position tensor,
            the selected batch tensor, and the updated local reference frames (LFrames) tensor.
        """
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.int64, device=pos.device)

        # get com of remaining points and choose as final center the point closest to com:
        coms = scatter(pos, batch, dim=0, reduce="mean")

        # distances to corresponding com:
        distances = torch.linalg.norm(pos - coms[batch], dim=-1)

        # get closest point:
        idx = scatter_min(distances, batch, dim=0)[1]

        col = torch.arange(len(pos), device=batch.device)
        row = batch  # picks the one center that is left for each batch.
        edge_index = torch.stack([col, row], dim=0)

        if self.use_skip_connections:
            x = torch.tensor([], device=pos.device) if x is None else x
            for layer_output in cached_layer_outputs[1:-1]:
                x = torch.cat([x, layer_output["x"]], dim=-1)

        x_dst = None if x is None else x[idx]
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


def lframes_knn_interpolate(
    x: torch.Tensor,
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    lframes_x: LFrames,
    lframes_y: LFrames,
    reps: Union[TensorReps, Irreps],
    batch_x: OptTensor = None,
    batch_y: OptTensor = None,
    k: int = 3,
    num_workers: int = 1,
):
    """The k-NN interpolation from the `"PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper (modified to handle local frames).
    For each point :math:`y` with position :math:`\\mathbf{p}(y)`, its
    interpolated features :math:`\\mathbf{f}(y)` are given by

    .. math::
        \\mathbf{f}(y) = \frac{\\sum_{i=1}^k w(x_i) \\mathbf{f}(x_i)}{\\sum_{i=1}^k
        w(x_i)} \textrm{, where } w(x_i) = \frac{1}{d(\\mathbf{p}(y),
        \\mathbf{p}(x_i))^2}

    and :math:`\\{ x_1, \\ldots, x_k \\}` denoting the :math:`k` nearest points
    to :math:`y`.

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\\mathbf{X} \\in \\mathbb{R}^{N \times F}`.
        pos_x (torch.Tensor): Node position matrix
            :math:`\\in \\mathbb{R}^{N \times d}`.
        pos_y (torch.Tensor): Upsampled node position matrix
            :math:`\\in \\mathbb{R}^{M \times d}`.
        lframes_x (LFrames): Local frames of the input nodes.
        lframes_y (LFrames): Local frames of the upsampled nodes.
        batch_x (torch.Tensor, optional): Batch vector
            :math:`\\mathbf{b_x} \\in {\\{ 0, \\ldots, B-1\\}}^N`, which assigns
            each node from :math:`\\mathbf{X}` to a specific example.
            (default: `None`)
        batch_y (torch.Tensor, optional): Batch vector
            :math:`\\mathbf{b_y} \\in {\\{ 0, \\ldots, B-1\\}}^N`, which assigns
            each node from :math:`\\mathbf{Y}` to a specific example.
            (default: `None`)
        k (int, optional): Number of neighbors. (default: `3`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case `batch_x` or `batch_y` is not
            `None`, or the input lies on the GPU. (default: `1`)
    """

    with torch.no_grad():
        assign_index = knn(
            pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y, num_workers=num_workers
        )
        y_idx, x_idx = (
            assign_index[0],
            assign_index[1],
        )  # y index are the receivers, x index are the senders
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-6)

    lframes_x = lframes_x.index_select(x_idx)
    lframes_y = lframes_y.index_select(y_idx)
    U = ChangeOfLFrames(lframes_start=lframes_x, lframes_end=lframes_y)
    transform = reps.get_transform_class()
    x_in_lframe_y = transform(x[x_idx], U)
    y = scatter(x_in_lframe_y * weights, y_idx, 0, pos_y.size(0), reduce="sum")
    y = y / scatter(weights, y_idx, 0, pos_y.size(0), reduce="sum")

    return y


class FPModule(torch.nn.Module):
    """Feature Propagation Module (FPModule) for PointNet architecture.

    Attributes:
        mlp (MLPWrapped): Multi-Layer Perceptron module.
        k (int): Number of nearest neighbors to consider during interpolation.
        reps (Union[TensorReps, Irreps]): Tensor representations or irreducible representations.
        out_dim (int): Output dimension of the MLP module.
        lframes_updater (QuaternionsUpdateLFrames | None): Local frames updater module.
    """

    def __init__(
        self,
        mlp: MLPWrapped,
        reps: Union[TensorReps, Irreps],
        k: int = 3,
        lframes_updater: QuaternionsUpdateLFrames | None = None,
    ) -> None:
        """Initializes a new instance of the FPModule class.

        Args:
            mlp (MLPWrapped): Multi-Layer Perceptron module.
            reps (Union[TensorReps, Irreps]): Tensor representations or irreducible representations.
            k (int, optional): Number of nearest neighbors to consider during interpolation. Defaults to 3.
            lframes_updater (QuaternionsUpdateLFrames | None, optional): Local frames updater module. Defaults to None.
        """
        super().__init__()
        self.mlp = mlp
        self.k = k
        self.reps = reps
        self.out_dim = mlp.out_dim
        self.lframes_updater = lframes_updater

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        lframes: LFrames,
        x_skip: torch.Tensor | None,
        pos_skip: torch.Tensor,
        batch_skip: torch.Tensor,
        lframes_skip: LFrames,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, LFrames]:
        """Forward pass of the FPModule.

        Args:
            x (torch.Tensor): Input tensor.
            pos (torch.Tensor): Position tensor.
            batch (torch.Tensor): Batch tensor.
            lframes (LFrames): Local frames tensor.
            x_skip (torch.Tensor): Skip connection input tensor.
            pos_skip (torch.Tensor): Skip connection position tensor.
            batch_skip (torch.Tensor): Skip connection batch tensor.
            lframes_skip (LFrames): Skip connection local frames tensor.

        Returns:
            torch.Tensor: Output features.
            torch.Tensor: Output position tensor.
            torch.Tensor: Output batch tensor.
            Lframes: Output local frames object.
        """
        x = lframes_knn_interpolate(
            x, pos, pos_skip, lframes, lframes_skip, self.reps, batch, batch_skip, k=self.k
        )
        if x_skip is not None:
            # this step is compatible with local frames (since the skip comes from the same frame)
            x = torch.cat([x, x_skip], dim=-1)  # expect last dim to be the feature dim

        if self.lframes_updater is not None:
            x, lframes_skip = self.lframes_updater(lframes=lframes_skip, x=x, batch=batch)
        x = self.mlp(x, batch=batch)
        return x, pos_skip, batch_skip, lframes_skip


class FinalLframesLayer(torch.nn.Module):
    """Final L-frames Layer module for PointNet architecture."""

    def __init__(
        self,
        in_channels: int,
        out_reps: Union[TensorReps, Irreps],
        mlp_channels: list[int] = None,
        final_activation: torch.nn.Module = None,
        **mlp_kwargs,
    ) -> None:
        """Initializes the FinalLframesLayer module.

        Args:
            in_channels (int): Number of input channels.
            out_reps (Union[TensorReps, Irreps]): Output representations.
            mlp_channels (list[int], optional): Hidden and output channels of the MLP. If None, no MLP is used. Defaults to None.
            final_activation (torch.nn.Module, optional): Final activation function. Defaults to None.
            **mlp_kwargs: Additional keyword arguments for the MLP module.
        """
        super().__init__()
        out_reps = out_reps
        if mlp_channels is None:
            self.mlp = None
            self.linear = torch.nn.Linear(in_channels, out_reps.dim)
        else:
            self.mlp = MLPWrapped(
                in_channels=in_channels, hidden_channels=mlp_channels, **mlp_kwargs
            )
            self.linear = torch.nn.Linear(mlp_channels[-1], out_reps.dim)
        self.final_activation = final_activation

    def forward(self, x, pos, batch, lframes):
        """Forward pass of the FinalLframesLayer module.

        Args:
            x (torch.Tensor): Input tensor.
            pos (torch.Tensor): Position tensor.
            batch (torch.Tensor): Batch tensor.
            lframes (LFrames): Local frames tensor.

        Returns:
            torch.Tensor: Output feature tensor.
            torch.Tensor: Output position tensor.
            torch.Tensor: Output batch tensor.
            LFrames: Output local frames object.
        """
        if self.mlp is not None:
            x = self.mlp(x, batch=batch)
        x = self.linear(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x, pos, batch, lframes
