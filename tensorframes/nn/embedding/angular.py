from typing import Tuple, Union

import e3nn.o3 as o3
import numpy as np
import torch
from torch import Tensor

from tensorframes.lframes import LFrames
from tensorframes.nn.embedding.radial import (
    compute_edge_vec,
    double_gradient_safe_normalize,
)


class AngularEmbedding(torch.nn.Module):
    """Angular Embedding module."""

    def __init__(self, out_dim: int) -> None:
        """Initializes an instance of the AngularEmbedding class.

        Args:
            out_dim (int): The output dimension of the embedding.
        """
        super().__init__()
        self.out_dim = out_dim  # should be set in the subclass

    def compute_embedding(self, edge_vec: Tensor) -> Tensor:
        """Computes the embedding for the given edge vector.

        Args:
            edge_vec (Tensor): The input edge vector.

        Returns:
            Tensor: The computed embedding.
        """
        raise NotImplementedError

    def forward(
        self,
        pos: Union[Tensor, Tuple] | None = None,
        edge_index: Tensor | None = None,
        lframes: LFrames | None = None,
        edge_vec: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the AngularEmbedding module.

        Either pos, edge_index, and lframes  or edge_vec must be provided.

        Args:
            pos (Union[Tensor, Tuple], optional): The position tensor or tuple.
            edge_index (Tensor, optional): The edge index tensor.
            lframes (LFrames, optional): The LFrames object. Defaults to None.
            edge_vec (Tensor, optional): The edge vector tensor. Defaults to None.

        Returns:
            Tensor: The computed embedding.
        """
        if edge_vec is None:
            assert lframes is not None, "lframes must be provided if edge_vec is not provided."
            assert pos is not None, "pos must be provided if edge_vec is not provided."
            assert (
                edge_index is not None
            ), "edge_index must be provided if edge_vec is not provided."
            edge_vec = compute_edge_vec(pos, edge_index, lframes=lframes)

        return self.compute_embedding(edge_vec)


class TrivialAngularEmbedding(AngularEmbedding):
    """A trivial implementation of the AngularEmbedding class."""

    def __init__(self, normalize: bool = True) -> None:
        """Init TrivialAngularEmbedding module.

        Args:
            normalize (bool, optional): Indicates whether to normalize the computed embeddings. Defaults to True.

        Attributes:
            normalize (bool): Indicates whether to normalize the computed embeddings.
            out_dim (int): The output dimension of the embeddings.
        """
        super().__init__(out_dim=3)
        self.normalize = normalize

    def compute_embedding(self, edge_vec: Tensor) -> Tensor:
        """Computes the embedding for the given edge vector.

        Args:
            edge_vec (Tensor): The input edge vector.

        Returns:
            Tensor: The computed embedding.
        """
        if self.normalize:
            return double_gradient_safe_normalize(edge_vec)
        else:
            return edge_vec


class SphericalHarmonicsEmbedding(AngularEmbedding):
    """Spherical Harmonics Embedding module."""

    def __init__(self, lmax: int = 2, normalization: str = "norm") -> None:
        """Init Spherical Harmonics Embedding module.

        Args:
            lmax (int): Maximum degree of the spherical harmonics expansion. Default is 2.
            normalization (str): Normalization method for the spherical harmonics. Choose from ["component", "norm", "integral"].
                Default is "norm".
        """
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        super().__init__(out_dim=self.irreps_sh.dim - 1)
        self.normalization = normalization

    def compute_embedding(self, edge_vec: Tensor) -> Tensor:
        """Compute the spherical harmonics embedding for the given edge vectors.

        Args:
            edge_vec (Tensor): Tensor of shape (batch_size, num_edges, 3) representing the edge vectors.

        Returns:
            Tensor: Tensor of shape (batch_size, num_edges, out_dim) representing the computed embeddings.
        """
        return o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=True, normalization=self.normalization
        )[..., 1:]


def fibonacci_sphere(samples: int = 1000) -> list:
    """Generate points on a sphere using the Fibonacci sphere algorithm.

    Args:
        samples (int): The number of points to generate on the sphere. Default is 1000.

    Returns:
        list: A list of tuples representing the points on the sphere. Each tuple contains the x, y, and z coordinates.
    """
    points = []
    phi = np.pi * (np.sqrt(5.0) - 1.0)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return points


class GaussianOnSphereEmbedding(AngularEmbedding):
    """A class representing a Gaussian embedding on a sphere."""

    def __init__(
        self, num_angular: int, normalized: bool = True, is_learnable: bool = True
    ) -> None:
        """Initialize the AngularEmbedding layer.

        Args:
            num_angular (int): The number of angular points to generate on the sphere.
            normalized (bool, optional): Whether to normalize the angular embeddings. Defaults to True.
            is_learnable (bool, optional): Whether the angular embeddings are learnable parameters. Defaults to True.
        """
        super().__init__(out_dim=num_angular)
        self.normalized = normalized

        # get num_angular equi spaced points on the sphere:
        points = torch.tensor(fibonacci_sphere(num_angular))

        # calculate mean of the difference of the points
        mean = torch.mean(torch.norm(points[1:] - points[:-1]), dim=0)
        widths = mean.repeat(num_angular).view(1, num_angular)

        # initialize the gaussian widths to be the mean of the difference of the points
        if is_learnable:
            self.centers = torch.nn.Parameter(data=points)
            self.widths = torch.nn.Parameter(data=widths)
        else:
            self.register_buffer("centers", points)
            self.register_buffer("widths", widths)

    def compute_embedding(self, edge_vec: Tensor) -> Tensor:
        """Compute the Gaussian embedding for the given edge vector.

        Args:
            edge_vec (Tensor): The input edge vector.

        Returns:
            Tensor: The computed Gaussian embedding.
        """
        edge_vec = double_gradient_safe_normalize(edge_vec)
        diff_norm = torch.linalg.norm(edge_vec.unsqueeze(1) - self.centers, dim=-1)

        # calculate the gaussian embedding
        exp = torch.exp(-0.5 * (diff_norm / self.widths) ** 2).to(edge_vec.dtype)

        # normalize the gaussians
        if self.normalized:
            exp = exp / torch.sqrt(self.widths * torch.pi).to(edge_vec.dtype)

        return exp
