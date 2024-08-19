from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from tensorframes.lframes import LFrames
from tensorframes.reps.tensorreps import TensorReps


def double_gradient_safe_norm(edge_vec: Tensor, eps: float = 1e-6) -> Tensor:
    """Needed when edge_vec contains zero vectors and when differentiating twice."""
    non_zero_mask = edge_vec.abs().sum(dim=-1) > eps
    norm = torch.zeros(edge_vec.shape[0], 1, device=edge_vec.device)
    norm[non_zero_mask] = torch.linalg.norm(edge_vec[non_zero_mask], dim=-1, keepdim=True)
    return norm


def double_gradient_safe_normalize(edge_vec: Tensor, eps: float = 1e-6) -> Tensor:
    """Needed when edge_vec contains zero vectors and when differentiating twice."""
    non_zero_mask = edge_vec.abs().sum(dim=-1) > eps
    norm = torch.linalg.norm(edge_vec[non_zero_mask], dim=-1, keepdim=True)
    edge_vec[non_zero_mask] = edge_vec[non_zero_mask] / norm
    return edge_vec


def compute_edge_vec(
    pos: Union[Tensor, Tuple], edge_index: Tensor, lframes: Union[LFrames, Tuple] | None = None
) -> Tensor:
    """Compute the edge vectors between node positions and rotates them into the local frames of
    the receiving nodes.

    Args:
        pos (Union[Tensor, Tuple]): The positions of the atoms. Can be a single tensor or a tuple of tensors.
        edge_index (Tensor): The edge index representing the connections between atoms.
        lframes (Union[Tensor, Tuple], optional): The local frames of the atoms. Can be a single tensor or a tuple of tensors. Defaults to None.

    Returns:
        Tensor: The computed edge vectors.
    """

    # get the position of the atom i and j from the edge index
    # j is the first row of the edge index and i is the second row
    if not isinstance(pos, tuple):
        pos_i = pos.index_select(0, edge_index[1])
        pos_j = pos.index_select(0, edge_index[0])
    else:
        pos_i = pos[1].index_select(0, edge_index[1])
        pos_j = pos[0].index_select(0, edge_index[0])

    if not isinstance(lframes, tuple):
        lframes = (lframes, lframes)

    edge_vec = pos_j - pos_i
    if lframes[1] is not None:
        rep_trafo = TensorReps("1x1n").get_transform_class()
        edge_vec = rep_trafo.forward(edge_vec, lframes[1].index_select(edge_index[1]))

    return edge_vec


class RadialEmbedding(Module):
    """RadialEmbedding module for computing embeddings based on radial distances.

    Methods:
        compute_embedding(norm: Tensor) -> Tensor:
            Computes the embedding based on the given norm.

        forward(pos: Union[Tensor, Tuple], edge_index: Tensor, edge_vec: Tensor = None) -> Tensor:
            Forward pass of the RadialEmbedding module.
    """

    def __init__(self, out_dim: int):
        """Initializes a RadialEmbeddingLayer object.

        Args:
            out_dim (int): The output dimension of the embedding layer.
        """
        super().__init__()
        self.out_dim = out_dim

    def compute_embedding(self, norm: Tensor) -> Tensor:
        """Computes the embedding based on the given norm.

        Args:
            norm (Tensor): The norm of the distance vector. Shape: E x 1.

        Returns:
            Tensor: The computed embedding. of shape E x out_dim.
        """
        raise NotImplementedError

    def forward(
        self,
        pos: Union[Tensor, Tuple] | None = None,
        edge_index: Tensor | None = None,
        edge_vec: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the RadialEmbedding module.

        Either pos and edge_index or edge_vec must be provided.

        Args:
            pos (Union[Tensor, Tuple], optional): The position tensor or tuple.
            edge_index (Tensor, optional): The edge index tensor.
            edge_vec (Tensor, optional): The edge vector tensor. Defaults to None.

        Returns:
            Tensor: The computed embedding.
        """
        if edge_vec is None:
            assert pos is not None, "pos must be provided if edge_vec is not provided."
            assert (
                edge_index is not None
            ), "edge_index must be provided if edge_vec is not provided."
            edge_vec = compute_edge_vec(pos, edge_index)

        # calculate the norm of the distance vector
        # norm = torch.linalg.norm(edge_vec, dim=-1, keepdim=True)  # Shape E x 1
        norm = double_gradient_safe_norm(edge_vec)  # Shape E x 1

        return self.compute_embedding(norm)


class BesselEmbedding(RadialEmbedding):
    """BesselEmbedding class represents a radial embedding using Bessel functions."""

    def __init__(
        self,
        num_frequencies: int,
        cutoff: float | None = None,
        envelope: Module | None = None,
        flip_negative: bool = False,
    ) -> None:
        """Initialize the RadialEmbedding layer.

        Args:
            num_frequencies (int): The number of frequencies to use.
            cutoff (float, optional): The cutoff value for the envelope function. Defaults to None.
            envelope (Module, optional): The envelope function to use. Defaults to None.
            flip_negative (bool, optional): Whether to flip the negative frequencies. Defaults to False.
        """
        super().__init__(out_dim=num_frequencies)

        self.num_frequencies = num_frequencies
        self.envelope = envelope
        self.flip_negative = flip_negative

        if cutoff is not None and self.envelope is not None:
            self.inv_cutoff = 1 / cutoff
            self.norm_const = (2 * self.inv_cutoff) ** 0.5
        else:
            self.envelope = None

        data = torch.pi * torch.arange(1, num_frequencies + 1)
        # Initialize frequencies at canonical positions
        self.frequencies = torch.nn.Parameter(
            data=data,
            requires_grad=True,
        )

    def compute_embedding(self, norm: Tensor) -> Tensor:
        """Computes the embedding for the given norm.

        Args:
            norm (Tensor): The norm of the input.

        Returns:
            Tensor: The computed embedding.
        """
        if self.envelope is None:
            out = torch.sin(norm * self.frequencies) / (norm + 1e-9)
        else:
            norm_scaled = norm * self.inv_cutoff
            norm_env = self.envelope(norm_scaled)
            out = (
                norm_env
                * self.norm_const
                * torch.sin(norm_scaled * self.frequencies)
                / (norm + 1e-9)
            )

        if self.flip_negative:
            out = torch.where(norm < 0, -out, out)

        return out


def get_gaussian_width(
    num_gaussians: int, maximum_initial_radius: float, intersection: float = 0.5
) -> float:
    """Calculate the width of the gaussian functions based on the number of gaussians and the
    maximum initial radius of the gaussians such that neighboring gaussians have an specified
    intersection.

    Args:
        num_gaussians (int): Number of gaussian functions on which the edge features are calculated.
        maximum_initial_radius (float): The maximum initial radius of the gaussians.
        intersection (float, optional): The intersection of the gaussians. Defaults to 0.5.

    Returns:
        float: The width of the gaussian functions.
    """
    if num_gaussians == 1:
        return maximum_initial_radius / np.sqrt(-2 * np.log(intersection))
    else:
        diff = maximum_initial_radius / (num_gaussians - 1)
        return diff / np.sqrt(-8 * np.log(intersection))


class GaussianEmbedding(RadialEmbedding):
    """Module to calculate the edge attributes based on gaussian basis functions and the distance
    between nodes."""

    def __init__(
        self,
        num_gaussians: int = 10,
        normalized: bool = False,
        maximum_initial_range: float = 5.0,
        minimum_initial_range: float = 0.0,
        is_learnable: bool = True,
        intersection: float = 0.5,
        gaussian_width: float | None = None,
    ) -> None:
        """Initialises the class. You can specify the number of gaussians and if the gaussians
        should be normalized. This function initialises the shift and scale parameters of the
        gaussians as learnable parameters.

        Args:
            num_gaussians (int, optional): Number of gaussian functions on which the edge features are calculated.
                Defaults to 10.
            normalized (bool, optional): Defines if the gaussians should be normalized. Defaults to False.
            maximum_initial_radius (float, optional): The maximum initial radius of the gaussians. Defaults to 5.0.
            is_learnable (bool, optional): Defines if the parameters of the gaussians should be learnable. Defaults to True.
            intersection (float, optional): The intersection of the gaussians, used to compute the width if not specified. Defaults to 0.5.
            gaussian_width (float, optional): The width of the gaussian functions. If None it calculates the gaussian width. Defaults to None.
        """
        super().__init__(out_dim=num_gaussians)

        self.num_gaussians = num_gaussians

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
            self.register_buffer(
                "shift",
                torch.linspace(minimum_initial_range, maximum_initial_range, num_gaussians),
            )

        self.normalized = normalized

    def compute_embedding(self, norm: Tensor) -> Tensor:
        """Calculates the embedding of the edge attributes based on the gaussian functions.

        .. math::
            e_{ij}^{k} = \\exp\\left(-\frac{1}{2} \\left(\frac{\\|r_i - r_j\\| - \\mu^k}
            {\\sigma^k} \right)^2\right).

        Args:
            pos (Tensor): positions of the atoms
            edge_index (Tensor): edge_index of the graph to calculate the edge attributes

        Returns:
            Tensor: The edge attributes. Shape: (E, num_gauss)
        """
        squared_diff = torch.square(norm - self.shift)
        squared_scale = torch.square(self.scale)

        # calculate the gaussian
        gaussian = torch.exp(-squared_diff / (2 * squared_scale))

        # if the gaussians should be normalized, divide by the normalization factor
        if self.normalized:
            gaussian = gaussian / (np.sqrt(2 * np.pi) * self.scale)

        return gaussian


class TrivialRadialEmbedding(RadialEmbedding):
    """A trivial radial embedding class that returns the norm of an edge vector as the
    embedding."""

    def __init__(self) -> None:
        """Initialises the class."""
        super().__init__(out_dim=1)

    def compute_embedding(self, norm: Tensor) -> Tensor:
        """Computes the embedding for the given input tensor.

        Args:
            norm (Tensor): The input tensor.

        Returns:
            Tensor: The computed embedding, which is the same as the input tensor.
        """
        return norm
