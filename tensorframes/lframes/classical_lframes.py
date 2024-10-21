import torch
from e3nn.o3 import rand_matrix
from torch import Tensor
from torch_geometric.nn import knn

from tensorframes.lframes.gram_schmidt import gram_schmidt
from tensorframes.lframes.lframes import LFrames


class ThreeNNLFrames(torch.nn.Module):
    """Computes local frames using the 3-nearest neighbors.

    The Frames are O(3) equivariant.
    """

    def __init__(self) -> None:
        """Initializes an instance of the ThreeNNLFrames class."""
        super().__init__()

    def forward(
        self, pos: Tensor, idx: Tensor | None = None, batch: Tensor | None = None
    ) -> LFrames:
        """Forward pass of the LFrames module.

        Args:
            pos (Tensor): The input tensor of shape (N, D) representing the positions of N points in D-dimensional space.
            idx (Tensor, optional): The indices of the points to consider. If None, all points are considered. Defaults to None.
            batch (Tensor, optional): The batch indices of the points. If None, a batch of zeros is used. Defaults to None.

        Returns:
            LFrames: The computed local frames as an instance of the LFrames class.
        """

        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.int64, device=pos.device)

        if idx is None:
            idx = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)

        # convert idx to bool tensor:
        if idx.dtype != torch.bool:
            idx = torch.zeros(
                pos.shape[0], dtype=torch.bool, device=pos.device
            ).scatter_(0, idx, True)

        # find 2 closest neighbors:
        row, col = knn(pos, pos[idx], k=4, batch_x=batch, batch_y=batch[idx])
        mask_self_loops = (
            torch.arange(pos.shape[0], dtype=int, device=idx.device)[idx][row] == col
        )
        assert (
            mask_self_loops.sum() == idx.sum()
        ), f"every center should have a self loop, {mask_self_loops.sum()} != {idx.sum()}"
        row = row[~mask_self_loops]
        col = col[~mask_self_loops].reshape((-1, 3))  # remove self loops

        assert torch.all(row[1:] >= row[:-1]), "row must be sorted"

        # compute the local frames:
        row = torch.arange(col.shape[0], device=pos.device)
        x_axis = pos[col[:, 0]] - pos[row]
        y_axis = pos[col[:, 1]] - pos[row]
        z_axis = pos[col[:, 2]] - pos[row]

        matrices = gram_schmidt(x_axis, y_axis, z_axis)

        return LFrames(matrices)


class RandomLFrames(torch.nn.Module):
    """Randomly generates local frames for each node."""

    def __init__(self, flip_probability=0.5) -> None:
        """Initialize an instance of the RandomLFrames class.

        Args:
            flip_probability (float, optional): The probability of flipping the frames. Defaults to 0.5.
        """
        super().__init__()
        self.flip_probability = flip_probability

    def forward(
        self, pos: Tensor, idx: Tensor | None = None, batch: Tensor | None = None
    ) -> LFrames:
        """Forward pass of the LFrames module.

        Args:
            pos (Tensor): The input tensor representing the positions.
            idx (Tensor | None, optional): The indices to select from the input tensor. Defaults to None.
            batch (Tensor | None, optional): The batch tensor. Defaults to None.

        Returns:
            LFrames: The output LFrames.
        """
        if idx is None:
            idx = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
        lframes = rand_matrix(pos[idx].shape[0], device=pos.device)
        if self.flip_probability > 0:
            flip_mask = (
                torch.rand(lframes.shape[0], device=lframes.device)
                < self.flip_probability
            )
            # flip the x-axis
            lframes[flip_mask, 0] = -lframes[flip_mask, 0]
        return LFrames(lframes)


class RandomGlobalLFrames(torch.nn.Module):
    """Randomly generates a global frame."""

    def __init__(self) -> None:
        """Initializes an instance of the RandomGlobalLFrames class."""
        super().__init__()

    def forward(
        self, pos: Tensor, idx: Tensor | None = None, batch: Tensor | None = None
    ) -> LFrames:
        """Applies forward pass of the LFrames module.

        Args:
            pos (Tensor): The input tensor representing the positions.
            idx (Tensor | None, optional): The indices tensor. Defaults to None.
            batch (Tensor | None, optional): The batch tensor. Defaults to None.

        Returns:
            LFrames: The output LFrames tensor.
        """
        if idx is None:
            idx = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)

        # randomly generate one local frame
        matrix = rand_matrix(1, device=pos.device)

        # if random number is less than 0.5, flip the x-axis
        if torch.rand(1, device=pos.device) < 0.5:
            matrix[0] = -matrix[0]

        return LFrames(matrix.repeat(pos[idx].shape[0], 1, 1))


class IdentityLFrames(torch.nn.Module):
    """Identity local frames."""

    def __init__(self) -> None:
        """Initializes an instance of the ClassicalLFrames class."""
        super().__init__()

    def forward(
        self, pos: Tensor, idx: Tensor | None = None, batch: Tensor | None = None
    ) -> LFrames:
        """Forward pass of the LFrames module.

        Args:
            pos (Tensor): The input tensor of shape (N, 4) representing the positions.
            idx (Tensor | None): The index tensor of shape (N,) representing the indices to select from `pos`.
                If None, all indices are selected.
            batch (Tensor | None): The batch tensor of shape (N,) representing the batch indices.

        Returns:
            LFrames: The output LFrames object.
        """
        if idx is None:
            idx = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)

        return LFrames(torch.eye(4, device=pos.device).repeat(pos[idx].shape[0], 1, 1))
