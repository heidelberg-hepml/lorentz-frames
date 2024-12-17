import torch
import numpy as np


class sampleLorentz:
    """
    Sampler class for random lorentz transformations
    """

    def __init__(
        self,
        trafo_types: list = ["rot", "boost"],
        axes: list = [[1, 2], [0, 3]],
        std_eta: float = 1,
    ):
        """
        Initializer for Sampler

        Args:
            trafo_types (list): List of trafo_types in order, eg. ["rot", "boost", "rot"]
            axes (list): List of rotation / boost axes in order eg. [[1,2],[0,1],[1,3]]
<<<<<<< HEAD
            mean_eta (float): mean value for rapidity, used for sampling from normal distribution, defaults to 0
            std_eta (float): std of rapidity, used for sampling from normal distribution, defaults to 1
=======
            std_eta (float): std of rapidity, used for sampleing from normal distribution, defaults to 1
>>>>>>> 24187c8 (Hard-coded mean_eta to zero for now)
        """
        self.trafo_types = np.array(trafo_types)
        self.num_boosts = (self.trafo_types == "boost").sum()
        self.num_rots = (self.trafo_types == "rot").sum()
        self.axes = axes
        self.std_eta = std_eta
        self.num_trafo = len(trafo_types)

        assert (
            len(trafo_types) == self.num_boosts + self.num_rots
        ), "trafo_types expect 'rot' or 'boosts', but got other input"

    def rand_matrix(self, N: int = 1, device: str = "cpu"):
        """
        Sample N transformation matrices with the parameters from the constructor

        Args:
            N (int): Number of the transformation matrices to sample

        Returns:
            final_trafo (tensor): tensor of stacked transformation matrices, shape: (batch, 4, 4)
        """
        angles = torch.empty((N, len(self.trafo_types)), device=device)

        angles[:, self.trafo_types == "boost"] = (
            torch.randn((N, self.num_boosts), device=device) * self.std_eta
        )
        angles[:, self.trafo_types == "rot"] = (
            torch.rand((N, self.num_rots), device=device) * 2 * torch.pi
        )

        return self.matrix(N=N, angles=angles, device=device)

    def matrix(self, N: int = 1, angles: torch.Tensor = None, device: str = "cpu"):
        """
        Create transformation matrices with given angles

        Args:
            N (int): Number of the transformation matrices to create / number of independent transformed systems i.e. graphs / jets
            angles (torch.tensor): angles to be used for matrices, shape: (N, num_trafos)

        Returns:
            final_trafo (torch.Tensor): transformation tensors of shape (N, 4, 4)
        """
        if not isinstance(angles, torch.Tensor):
            angles = torch.tensor(angles)
        assert (
            angles.shape[0] == N
        ), f"Need angles for all matrices, but got {angles.shape[0]} instead of {N}!"

        final_trafo = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)
        for trafo_type, axes, angle in zip(self.trafo_types, self.axes, angles.T):
            if trafo_type == "boost":
                temp = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)
                temp[:, axes[0], axes[0]] = torch.cosh(angle)
                temp[:, axes[0], axes[1]] = torch.sinh(angle)
                temp[:, axes[1], axes[0]] = torch.sinh(angle)
                temp[:, axes[1], axes[1]] = torch.cosh(angle)
                final_trafo = torch.einsum("ijk,ikl->ijl", temp, final_trafo)
            elif trafo_type == "rot":
                temp = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)
                temp[:, axes[0], axes[0]] = torch.cos(angle)
                temp[:, axes[0], axes[1]] = -torch.sin(angle)
                temp[:, axes[1], axes[0]] = torch.sin(angle)
                temp[:, axes[1], axes[1]] = torch.cos(angle)
                final_trafo = torch.einsum("ijk,ikl->ijl", temp, final_trafo)
            else:
                assert (
                    False
                ), f"Expected trafo_types 'rot' or 'boost', but got {trafo_type} instead."
        return final_trafo.to(device)
