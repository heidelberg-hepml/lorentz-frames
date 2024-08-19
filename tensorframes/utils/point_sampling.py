import torch
import torch_geometric as tg

from tensorframes.utils import batch_to_ptr


class RandomSampler(torch.nn.Module):
    """Randomly samples points from a given input tensor."""

    def __init__(
        self, ratio: float, seed: int | None = None, with_replacement: bool = False
    ) -> None:
        """Initialize the PointSampler object.

        Args:
            ratio (float): The ratio of points to sample.
            seed (int | None, optional): The seed value for random number generation. Defaults to None.
            with_replacement (bool, optional): Whether to sample points with replacement. Defaults to False.
        """
        super().__init__()
        self.ratio = ratio
        self.seed = seed
        self.with_replacement = with_replacement

    def forward(
        self, pos: torch.Tensor, batch: torch.Tensor | None = None, ptr: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass of the RandomSampler.

        Args:
            pos (torch.Tensor): The input tensor containing the points.
            batch (torch.Tensor, optional): The batch tensor. Default is None.
            ptr (torch.Tensor, optional): The pointer tensor. Default is None.

        Returns:
            torch.Tensor: The indices of the sampled points.
        """
        if self.ratio == 1:
            return torch.arange(pos.shape[0], device=pos.device)

        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.int64)
        num_points = torch.bincount(batch)
        if ptr is None:
            ptr = batch_to_ptr(batch)
        if self.seed is not None:
            torch.manual_seed(self.seed)
        if self.with_replacement:
            idx = torch.concatenate(
                [
                    torch.randint(low=ptr[i], high=ptr[i + 1], size=(int(num * self.ratio),))
                    for i, num in enumerate(num_points)
                ],
                dim=0,
            )
        else:
            idx = torch.concatenate(
                [
                    torch.arange(ptr[i], ptr[i + 1])[torch.randperm(num)[: int(num * self.ratio)]]
                    for i, num in enumerate(num_points)
                ],
                dim=0,
            )

        if self.seed is not None:
            torch.seed()

        return idx.to(pos.device)


class FPSampler(torch.nn.Module):
    """FPSampler is a module for performing farthest point sampling on point clouds.

    Attributes:
        ratio (float): The ratio of points to sample from the input point cloud.
        random_start (bool): Whether to randomly select the starting point for sampling.
    """

    def __init__(self, ratio: float, random_start: bool = True) -> None:
        """Initialize the PointSampling class.

        Args:
            ratio (float): The ratio of points to sample.
            random_start (bool, optional): Whether to start sampling from a random point. Defaults to True.
        """
        super().__init__()
        self.ratio = ratio
        self.random_start = random_start

    def forward(self, pos: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        """Performs forward pass of the point sampling module.

        Args:
            pos (torch.Tensor): The input tensor of positions.
            batch (torch.Tensor, optional): The input tensor of batch indices. Defaults to None.

        Returns:
            torch.Tensor: The output tensor of sampled points.
        """
        if self.ratio == 1:
            return torch.arange(pos.shape[0], device=pos.device)
        return tg.nn.fps(x=pos, batch=batch, ratio=self.ratio, random_start=self.random_start)


class CustomPointSampler(torch.nn.Module):
    """CustomPointSampler is a PyTorch module for point sampling.

    Attributes:
        methods (list): List of implemented sampling methods.
    """

    methods = ["fps", "random"]

    def __init__(self, sampling_method: str = "fps", **kwargs: dict | None) -> None:
        """Initialize the CustomPointSampler class.

        Args:
            sampling_method (str): The sampling method to use. Options are "fps" (Farthest Point Sampling) and "random".
            **kwargs: Additional keyword arguments specific to the chosen sampling method.

        Raises:
            AssertionError: If the specified sampling method is not implemented.
        """
        super().__init__()
        assert (
            sampling_method in self.methods
        ), f"Sampling method {sampling_method} not implemented."
        kwargs = {} if kwargs is None else kwargs

        if sampling_method == "fps":
            self.sampler = FPSampler(**kwargs)
        elif sampling_method == "random":
            self.sampler = RandomSampler(**kwargs)
        else:
            pass

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the CustomPointSampler module.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The result of the forward pass.
        """
        return self.sampler(*args, **kwargs)
