import copy

import numpy as np
import torch
from torch import Tensor

from tensorframes.nn.embedding.angular import AngularEmbedding
from tensorframes.nn.embedding.radial import (
    RadialEmbedding,
    double_gradient_safe_normalize,
    get_gaussian_width,
)


class AxisWiseEmbeddingFromRadial(AngularEmbedding):
    """A class representing axis-wise embedding from radial embeddings.

    Attributes:
        normalize_edge_vec (bool): Whether to normalize the edge vectors.
        axis_specific_radial (bool): Whether to use axis-specific radial embeddings.
        radial_modules (Union[torch.nn.ModuleList, torch.nn.Module]): List of radial embedding modules.
        spatial_dim (int): The spatial dimension.
        out_dim (int): The output dimension of the embedding.

    Methods:
        compute_embedding(edge_vec: Tensor) -> Tensor: Computes the embedding for the given edge vectors.
    """

    def __init__(
        self,
        radial_embedding: RadialEmbedding,
        normalize_edge_vec: bool = True,
        axis_specific_radial: bool = False,
        spatial_dim=3,
    ):
        """Initialize the AxisWiseEmbeddingFromRadial module.

        Args:
            radial_type (str): The type of radial embedding to use.
            normalize_edge_vec (bool, optional): Whether to normalize the edge vectors. Defaults to True.
            axis_specific_radial (bool, optional): Whether to use axis-specific radial embeddings. Defaults to False.
            spatial_dim (int, optional): The spatial dimension. Defaults to 3.
            **radial_kwargs: Additional keyword arguments to be passed to the radial embedding modules.
        """
        super().__init__(out_dim=radial_embedding.out_dim * spatial_dim)

        self.normalize_edge_vec = normalize_edge_vec
        self.axis_specific_radial = axis_specific_radial
        self.spatial_dim = spatial_dim

        if self.axis_specific_radial:
            self.radial_modules = torch.nn.ModuleList()
            for i in range(spatial_dim):
                if i == 0:
                    self.radial_modules.append(radial_embedding)
                else:
                    self.radial_modules.append(copy.deepcopy(radial_embedding))
        else:
            self.radial_modules = radial_embedding

    def compute_embedding(self, edge_vec: Tensor) -> Tensor:
        """Computes the embedding for the given edge vectors.

        Args:
            edge_vec (Tensor): The edge vectors.

        Returns:
            Tensor: The computed embedding.
        """
        if self.normalize_edge_vec:
            edge_vec = double_gradient_safe_normalize(edge_vec)

        if self.axis_specific_radial:
            out = []
            for i in range(self.spatial_dim):
                out.append(
                    self.radial_modules[i]
                    .compute_embedding(edge_vec[:, i].view(-1, 1))
                    .reshape(-1, self.radial_modules[i].out_dim)
                )
            return torch.cat(out, dim=-1)
        else:
            return self.radial_modules.compute_embedding(edge_vec.view(-1, 1)).reshape(
                -1, self.out_dim
            )


class AxisWiseBesselEmbedding(AngularEmbedding):
    """Embedding layer that computes axis-wise Bessel embeddings."""

    def __init__(
        self, num_frequencies: int, dual_sided: bool = True, is_learnable: bool = True
    ) -> None:
        """Initialize the AxisWiseBesselEmbedding module.

        Args:
            num_frequencies (int): The number of frequencies to use for the embeddings.
            dual_sided (bool, optional): Whether to use dual-sided embeddings. Defaults to True.
            is_learnable (bool, optional): Whether the frequencies are learnable parameters. Defaults to True.
        """
        super().__init__(out_dim=num_frequencies * 3)

        self.num_frequencies = num_frequencies

        data = torch.pi * torch.arange(1, num_frequencies + 1)

        data = data.repeat(3, 1)

        self.dual_sided = dual_sided

        # Initialize frequencies at canonical positions
        if is_learnable:
            self.frequencies = torch.nn.Parameter(data=data)
        else:
            self.register_buffer("frequencies", data)

    def compute_embedding(self, edge_vec: Tensor) -> Tensor:
        """Computes the embedding for the given edge vector.

        Args:
            edge_vec (Tensor): The input edge vector.

        Returns:
            Tensor: The computed embedding.
        """
        edge_vec = double_gradient_safe_normalize(edge_vec)
        tmp_mul = torch.einsum("ij,jk->ijk", edge_vec, self.frequencies)

        embed = torch.einsum(
            "ijk,ij->ijk", torch.sin(tmp_mul), 1 / (edge_vec + 1e-9)
        )  # shape E x 3

        if self.dual_sided:
            embed = torch.einsum("ijk, ij -> ijk", embed, torch.sign(edge_vec))

        out = embed.reshape(-1, self.num_frequencies * 3)

        return out


class AxisWiseGaussianEmbedding(AngularEmbedding):
    """Axis-wise Gaussian Embedding.

    This class represents an axis-wise Gaussian embedding for neural networks.

    Attributes:
        num_gaussians (int): The number of Gaussians used for the embedding.
        out_dim (int): The output dimension of the embedding.
        shift (torch.nn.Parameter or torch.Tensor): The shifts of the Gaussians.
        scale (torch.nn.Parameter or torch.Tensor): The scale parameters of the Gaussians.
        normalized (bool): Whether the Gaussians are normalized.

    Methods:
        compute_embedding(edge_vec: torch.Tensor) -> torch.Tensor:
            Computes the embedding for the given edge vector.
    """

    def __init__(
        self,
        num_gaussians: int = 10,
        normalized: bool = False,
        maximum_initial_range: float = 1.0,
        minimum_initial_range: float | None = None,
        is_learnable: bool = True,
        intersection: float = 0.5,
        gaussian_width: float | None = None,
    ) -> None:
        """Initialize the AxisWiseGaussianEmbedding module.

        Args:
            num_gaussians (int): The number of Gaussians to use for the embedding.
            normalized (bool): Whether to normalize the Gaussians.
            maximum_initial_range (float): The maximum initial range for the Gaussians.
            minimum_initial_range (float, optional): The minimum initial range for the Gaussians. If not provided, it is set to the negative value of `maximum_initial_range`.
            is_learnable (bool): Whether the embedding parameters are learnable.
            intersection (float): The intersection parameter for calculating the Gaussian width.
            gaussian_width (float, optional): The width of the Gaussians. If not provided, it is calculated based on the other parameters.
        """
        super().__init__(out_dim=num_gaussians * 3)

        self.num_gaussians = num_gaussians
        minimum_initial_range = (
            -maximum_initial_range if minimum_initial_range is None else minimum_initial_range
        )

        if gaussian_width is None:
            gaussian_width = get_gaussian_width(
                num_gaussians=num_gaussians,
                maximum_initial_radius=maximum_initial_range - minimum_initial_range,
                intersection=intersection,
            )
        if is_learnable:
            # use linspace to create the shifts of the gaussians in the range of 0 to 3
            # we use 3 as an initialisation because the bond length of c-c is 3 Bohr
            self.shift = torch.nn.Parameter(
                torch.linspace(minimum_initial_range, maximum_initial_range, num_gaussians)
            )
            # initialisation of the scale parameter as 1
            self.scale = torch.nn.Parameter(torch.ones(num_gaussians) * gaussian_width)
        else:
            self.register_buffer("scale", torch.ones(num_gaussians) * gaussian_width)
            self.register_buffer("shift", torch.linspace(0, maximum_initial_range, num_gaussians))

        self.normalized = normalized

    def compute_embedding(self, edge_vec: torch.Tensor) -> torch.Tensor:
        """Computes the embedding for the given edge vector.

        Args:
            edge_vec (torch.Tensor): The input edge vector.

        Returns:
            torch.Tensor: The computed embedding.
        """
        edge_vec = double_gradient_safe_normalize(edge_vec)
        squared_diff = torch.square(edge_vec.unsqueeze(2) - self.shift)
        squared_scale = torch.square(self.scale)

        # calculate the gaussian
        gaussian = torch.exp(-squared_diff / (2 * squared_scale))

        # if the gaussians should be normalized, divide by the normalization factor
        if self.normalized:
            gaussian = gaussian / (np.sqrt(2 * np.pi) * self.scale)

        return gaussian.reshape(-1, self.num_gaussians * 3)
