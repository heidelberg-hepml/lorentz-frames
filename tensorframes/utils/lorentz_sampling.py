import torch


class sample_lorentz:
    """
    Sampler class for random lorentz transformations
    """

    def __init__(
        self,
        trafo_types: list = ["rot", "boost"],
        axes: list = [[1, 2], [0, 3]],
        mean_eta: float = 0,
        std_eta: float = 1,
    ):
        """
        Initializer for Sampler

        Args:
            trafo_types (list): List of trafo_types in order, eg. ["rot", "boost", "rot"]
            axes (list): List of rotation / boost axes in order eg. [[1,2],[0,1],[1,3]]
            mean_eta (float): mean value for rapidity, used for sampling from normal distribution, defaults to 0
            std_eta (float): std of rapidity, used for sampleing from normal distribution, defaults to 1
        """
        self.trafo_types = trafo_types
        self.axes = axes
        self.mean_eta = torch.tensor(mean_eta).to(torch.float)
        self.std_eta = torch.tensor(std_eta).to(torch.float)
        self.num_trafo = len(trafo_types)

    def rand_matrix(self, N: int = 1, device: str = "cpu"):
        """
        Sample N transformation matrices with the parameters from the constructor

        Args:
            N (int): Number of the transformation matrices to sample

        Returns:
            final_trafo (tensor): tensor of stacked transformation matrices, shape: (batch, 4, 4)
        """
        final_trafo = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)
        for i in range(self.num_trafo):
            if self.trafo_types[i] == "boost":
                angle = torch.normal(
                    mean=self.mean_eta.repeat(N), std=self.std_eta.repeat(N)
                )
                temp = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)
                axes = self.axes[i]
                temp[:, axes[0], axes[0]] = torch.cosh(angle)
                temp[:, axes[0], axes[1]] = torch.sinh(angle)
                temp[:, axes[1], axes[0]] = torch.sinh(angle)
                temp[:, axes[1], axes[1]] = torch.cosh(angle)
                final_trafo = torch.einsum("ijk,ikl->ijl", temp, final_trafo)
            elif self.trafo_types[i] == "rot":
                angle = torch.rand(N) * 2 * torch.pi
                temp = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)
                axes = self.axes[i]
                temp[:, axes[0], axes[0]] = torch.cos(angle)
                temp[:, axes[0], axes[1]] = -torch.sin(angle)
                temp[:, axes[1], axes[0]] = torch.sin(angle)
                temp[:, axes[1], axes[1]] = torch.cos(angle)
                final_trafo = torch.einsum("ijk,ikl->ijl", temp, final_trafo)
            else:
                assert (
                    False
                ), f"Expected trafo_types 'rot' or 'boost', but got {self.trafo_types[i]} instead."
        return final_trafo.to(device)

    def matrix(self, N: int = 1, angles: torch.Tensor = None, device: str = "cpu"):
        """
        Create transformation matrices with given angles

        Args:
            N (int): Number of the transformation matrices to create
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
        for i in range(self.num_trafo):
            if self.trafo_types[i] == "boost":
                angle = angles[:, i]
                temp = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)
                axes = self.axes[i]
                temp[:, axes[0], axes[0]] = torch.cosh(angle)
                temp[:, axes[0], axes[1]] = torch.sinh(angle)
                temp[:, axes[1], axes[0]] = torch.sinh(angle)
                temp[:, axes[1], axes[1]] = torch.cosh(angle)
                final_trafo = torch.einsum("ijk,ikl->ijl", temp, final_trafo)
            elif self.trafo_types[i] == "rot":
                angle = angles[:, i]
                temp = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)
                axes = self.axes[i]
                temp[:, axes[0], axes[0]] = torch.cos(angle)
                temp[:, axes[0], axes[1]] = -torch.sin(angle)
                temp[:, axes[1], axes[0]] = torch.sin(angle)
                temp[:, axes[1], axes[1]] = torch.cos(angle)
                final_trafo = torch.einsum("ijk,ikl->ijl", temp, final_trafo)
            else:
                assert (
                    False
                ), f"Expected trafo_types 'rot' or 'boost', but got {self.trafo_types[i]} instead."
        return final_trafo.to(device)
