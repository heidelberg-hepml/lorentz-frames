import torch
from tensorframes.utils.lorentz_sampling import sample_lorentz
from torch import Tensor
from torch_geometric.nn import knn

from tensorframes.lframes.gram_schmidt import gram_schmidt
from tensorframes.lframes.lframes import LFrames
from tensorframes.utils.utils import stable_arctanh


class LFramesPredictionModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs) -> LFrames:
        raise NotImplementedError, "Subclasses must implement this method."


class NNLFrames(LFramesPredictionModule):
    """Computes local frames using the 3-nearest neighbors.

    The Frames are SO(1,3) equivariant.
    """

    def __init__(self) -> None:
        """Initializes an instance of the NNLFrames class."""
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

        # find 3+1 closest neighbors:
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

        vectors = torch.stack((x_axis, y_axis, z_axis), axis=-2)
        matrices = gram_schmidt(vectors)

        return LFrames(matrices)


class RandomLFrames(LFramesPredictionModule):
    """Randomly generates local frames for each node."""

    def __init__(self, flip_probability: float = 0.5) -> None:
        """Initialize an instance of the RandomLFrames class."""
        raise NotImplementedError
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
        lframes = sample_lorentz(pos[idx].shape[0], device=pos.device)
        if self.flip_probability > 0:
            flip_mask = (
                torch.rand(lframes.shape[0], device=lframes.device)
                < self.flip_probability
            )
            # flip the x-axis
            lframes[flip_mask, 0] = -lframes[flip_mask, 0]
        return LFrames(lframes)


class RandomGlobalLFrames(LFramesPredictionModule):
    """Randomly generates a global frame."""

    def __init__(self, mean_eta, std_eta) -> None:
        """Initializes an instance of the RandomGlobalLFrames class."""
        super().__init__()
        self.sampler = sample_lorentz(mean_eta=mean_eta, std_eta=std_eta)

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
        matrix = self.sampler.rand_matrix(1, device=pos.device)

        # if random number is less than 0.5, flip the x-axis
        if torch.rand(1, device=pos.device) < 0.5:
            matrix[0] = -matrix[0]

        return LFrames.global_trafo(
            device=pos.device, trafo=matrix, n_batch=pos.shape[0]
        )


class IdentityLFrames(LFramesPredictionModule):
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

        return LFrames.global_trafo(pos.device, n_batch=pos.shape[0])


class COMLFrames(LFramesPredictionModule):
    """
    Creates a center-of-momentum frame for each jet, introducing a additional symmetry
    """

    def __init__(self):
        super().__init__()
        self.sampler = sample_lorentz(
            trafo_types=["rot", "rot", "boost"], axes=[[1, 2], [1, 3], [0, 1]]
        )

    def forward(
        self, pos: torch.Tensor, idx: torch.Tensor = None, batch: torch.Tensor = None
    ) -> LFrames:
        """
         Creates LFrames through transformation matrix into COM frame

        Args:
             pos (torch.Tensor): Tensor with all the positions of the nodes, shape=(batch, 4)
             idx (torch.Tensor): Index Tensor
             batch (torch.tensor): Batch tensor, shape=(batch)

         Return:
             Lframes of the graphs
        """

        if idx is not None:
            pos = pos[idx].clone()
            batch = batch[idx].clone()

        batchUni = torch.unique(batch)
        mean_pos = torch.empty(batchUni.max() + 1, 4, device=pos.device)
        for i in batchUni:  # this is inefficient but will work for now
            mean_pos[i] = torch.mean(pos[batch == i], axis=0)

        angles = torch.empty(batchUni.max() + 1, 3, device=pos.device)
        angles[:, 0] = -torch.arctan(mean_pos[:, 2] / mean_pos[:, 1])
        angles[:, 1] = -torch.arctan(
            mean_pos[:, 3]
            / (
                mean_pos[:, 1] * torch.cos(angles[:, 0])
                - mean_pos[:, 2] * torch.sin(angles[:, 0])
            )
        )
        angles[:, 2] = -stable_arctanh(
            torch.linalg.norm(mean_pos[:, 1:], dim=1) / mean_pos[:, 0]
        )

        trafo = self.sampler.matrix(
            N=batchUni.max() + 1, angles=angles, device=pos.device
        )
        # npos = torch.einsum("nmp,np->nm", trafo[batch], pos)

        return LFrames(trafo[batch])


class PartialCOMLFrames(LFramesPredictionModule):
    """
    Creates a center-of-momentum frame for each jet, however, restricting ourself to rotation the x-y momentum to be full aligned with the x axis and boosting the jet to remove the z-component
    """

    def __init__(self):
        super().__init__()
        self.sampler = sample_lorentz(
            trafo_types=["rot", "boost"], axes=[[1, 2], [0, 3]]
        )

    def forward(
        self, pos: torch.Tensor, idx: torch.Tensor = None, batch: torch.Tensor = None
    ) -> LFrames:
        """
        Creates LFrames through transformation matrix into partial COM frame

        Args:
            pos (torch.Tensor): Tensor with all the positions of the nodes, shape=(batch, 4)
            idx (torch.Tensor): Index Tensor
            batch (torch.tensor): Batch tensor, shape=(batch)

        Return:
            Lframes of the graphs
        """

        if idx is not None:
            pos = pos[idx].clone()
            batch = batch[idx].clone()

        batchUni = torch.unique(batch)
        mean_pos = torch.empty(batchUni.max() + 1, 4, device=pos.device)
        for i in batchUni:  # this is inefficient but will work for now
            mean_pos[i] = torch.mean(pos[batch == i], axis=0)

        angles = torch.empty(batchUni.max() + 1, 2, device=pos.device)
        angles[:, 0] = -torch.arctan(mean_pos[:, 2] / mean_pos[:, 1])
        angles[:, 1] = -stable_arctanh(mean_pos[:, 3] / mean_pos[:, 0])

        trafo = self.sampler.matrix(
            N=batchUni.max() + 1, angles=angles, device=pos.device
        )
        # npos = torch.einsum("nmp,np->nm", trafo[batch], pos)

        return LFrames(trafo[batch])


class RestLFrames(LFramesPredictionModule):
    """
    Creates a Rest frame for each particle in the batch, introducing an additional symmetry
    """

    def __init__(self):
        super().__init__()
        self.sampler = sample_lorentz(
            trafo_types=["rot", "rot", "boost"], axes=[[1, 2], [1, 3], [0, 1]]
        )

    def forward(
        self, pos: torch.Tensor, idx: torch.Tensor = None, batch: torch.Tensor = None
    ) -> LFrames:
        """
        Creates LFrames through transformation matrix into particle rest frame

        Args:
            pos (torch.Tensor): Tensor with all the positions of the nodes, shape=(batch, 4)
            idx (torch.Tensor): Index Tensor
            batch (torch.tensor): Batch tensor, shape=(batch)

        Return:
            Lframes of the graphs
        """

        if idx is not None:
            pos = pos[idx].clone()
            batch = batch[idx].clone()

        angles = torch.empty(len(pos), 3, device=pos.device)
        angles[:, 0] = -torch.arctan(pos[:, 2] / pos[:, 1])
        angles[:, 1] = -torch.arctan(
            pos[:, 3]
            / (
                pos[:, 1] * torch.cos(angles[:, 0])
                - pos[:, 2] * torch.sin(angles[:, 0])
            )
        )
        angles[:, 2] = -stable_arctanh(torch.linalg.norm(pos[:, 1:], dim=1) / pos[:, 0])

        trafo = self.sampler.matrix(N=len(pos), angles=angles, device=pos.device)
        # npos = torch.einsum("nmp,np->nm", trafo[batch], pos)

        return LFrames(trafo)


class PartialRestLFrames(LFramesPredictionModule):
    """
    Creates a Rest Frame for each particle, however, restricting ourself to rotation the x-y momentum to be full aligned with the x axis and boosting the jet to remove the z-component
    """

    def __init__(self):
        super().__init__()
        self.sampler = sample_lorentz(
            trafo_types=["rot", "boost"], axes=[[1, 2], [0, 3]]
        )

    def forward(
        self, pos: torch.Tensor, idx: torch.Tensor = None, batch: torch.Tensor = None
    ) -> LFrames:
        """
        Creates LFrames through transformation matrix into partial particle rest frame

        Args:
            pos (torch.Tensor): Tensor with all the positions of the nodes, shape=(batch, 4)
            idx (torch.Tensor): Index Tensor
            batch (torch.tensor): Batch tensor, shape=(batch)

        Return:
            Lframes of the graphs
        """

        if idx is not None:
            pos = pos[idx].clone()
            batch = batch[idx].clone()

        angles = torch.empty(len(pos), 2, device=pos.device)
        angles[:, 0] = -torch.arctan(pos[:, 2] / pos[:, 1])
        angles[:, 1] = -stable_arctanh(pos[:, 3] / pos[:, 0])

        trafo = self.sampler.matrix(N=len(pos), angles=angles, device=pos.device)
        # npos = torch.einsum("nmp,np->nm", trafo[batch], pos)

        return LFrames(trafo)
