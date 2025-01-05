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
            assert (
                lframes is not None
            ), "lframes must be provided if edge_vec is not provided."
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
        super().__init__(out_dim=4)
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
