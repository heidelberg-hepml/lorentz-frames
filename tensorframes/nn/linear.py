import math

import torch
from torch import Tensor
from torch.nn import Linear


class HeadedLinear(torch.nn.Module):
    """A linear layer with multiple heads.

    Every head has its own weight matrix and bias vector.
    """

    def __init__(
        self, in_dim: int, out_dim: int, num_heads: int, bias: bool = True
    ) -> None:
        """Initialize the Linear module.

        Args:
            in_dim (int): The input dimension.
            out_dim (int): The output dimension.
            num_heads (int): The number of heads.
            bias (bool, optional): Whether to include a bias vector.
        """
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.matrix = torch.nn.Parameter(torch.empty((num_heads, in_dim, out_dim)))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty((num_heads, out_dim)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters of the linear layer."""
        for i in range(self.num_heads):
            torch.nn.init.kaiming_uniform_(self.matrix[i], a=math.sqrt(5))
            if hasattr(self, "bias"):
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.matrix[i])
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass of the linear layer. The input tensor is multiplied by the
        weight matrix of each head.

        Args:
            x (Tensor): The input tensor. shape: (batch_size, num_heads, in_dim)

        Returns:
            Tensor: The output tensor after applying the linear transformation. shape: (batch_size, num_heads, out_dim)
        """
        out = torch.einsum("nhi, hio -> nho", x, self.matrix)

        if hasattr(self, "bias"):
            out += self.bias.unsqueeze(0)

        return out


class EdgeLinear(torch.nn.Module):
    """Calculates a linear transformation on the edge embeddings, is multiplied with the input
    tensor.

    And at the end a linear transformation is applied on the result.
    """

    def __init__(self, in_dim: int, emb_dim: int, out_dim: int) -> None:
        """Initialize the EdgeLinear module.

        Args:
            in_dim (int): The input dimension.
            emb_dim (int): The embedding dimension.
            out_dim (int): The output dimension.
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(emb_dim, in_dim)
        self.linear2 = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor, edge_embedding: Tensor) -> Tensor:
        """Performs the forward pass of the linear neural network layer.

        Args:
            x (Tensor): The input tensor.
            edge_embedding (Tensor): The edge embedding tensor.

        Returns:
            Tensor: The output tensor after applying the linear transformation.
        """
        edge = self.linear1(edge_embedding)

        if len(x.shape) == 3:
            out = self.linear2(torch.mul(x, edge.unsqueeze(1)))
        else:
            out = self.linear2(torch.mul(x, edge))
        return out


class HeadedEdgeLinear(torch.nn.Module):
    """Is a headed version of the EdgeLinear module.

    Every head has its own EdgeLinear module.
    """

    def __init__(
        self, in_dim: int, emb_dim: int, out_dim: int, num_heads: int, bias: bool = True
    ) -> None:
        """Initialize a HeadedEdgeLinear module.

        Args:
            in_dim (int): The input dimension.
            emb_dim (int): The embedding dimension.
            out_dim (int): The output dimension.
            num_heads (int): The number of heads.
            bias (bool, optional): Whether to include bias terms. Defaults to True.
        """
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.emb_dim = emb_dim

        self.lin_edge = Linear(emb_dim, in_dim * num_heads, bias)
        self.lin_out = HeadedLinear(in_dim, out_dim, num_heads, bias)

    def forward(self, x: Tensor, edge_embedding: Tensor) -> Tensor:
        """Performs the forward pass of the HeadedEdgeLinear module.

        Args:
            x (Tensor): The input tensor.
            edge_embedding (Tensor): The edge embedding tensor.

        Returns:
            Tensor: The output tensor after applying the linear transformation.
        """
        edge = self.lin_edge(edge_embedding).view(-1, self.num_heads, self.in_dim)
        x = torch.mul(x, edge)
        x = self.lin_out(x)
        return x
