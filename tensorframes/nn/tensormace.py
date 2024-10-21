import math
from typing import Union

import torch
from torch import Tensor
from torch_geometric.nn import LayerNorm, MessagePassing

from tensorframes.lframes.lframes import LFrames
from tensorframes.nn.linear import EdgeLinear
from tensorframes.reps.tensorreps import TensorReps


class TensorMACE(MessagePassing):
    """The TensorMACE model.

    TODO: Make it more verbose
    """

    def __init__(
        self,
        in_tensor_reps: TensorReps,
        out_tensor_reps: TensorReps,
        edge_emb_dim: int,
        hidden_dim: int,
        max_order: int = 3,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        """Initialize a TensorMace object.

        Args:
            in_tensor_reps (TensorReps): The input tensor representations.
            out_tensor_reps (TensorReps]): The output tensor representations.
            edge_emb_dim (int): The dimension of the edge embeddings.
            hidden_dim (int): The dimension of the hidden layer.
            order (int): The message passing body order. Defaults to 3.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            bias (bool, optional): Whether to include bias terms. Defaults to False.
        """
        super().__init__(
            params_dict={
                "x": {"type": "local", "rep": in_tensor_reps},
            }
        )
        self.in_tensor_reps = in_tensor_reps
        self.out_tensor_reps = out_tensor_reps
        self.in_dim = in_tensor_reps.dim
        self.out_dim = out_tensor_reps.dim
        self.edge_emb_dim = edge_emb_dim
        self.max_order = max_order
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.edge_lin = EdgeLinear(self.in_dim, self.edge_emb_dim, self.hidden_dim)

        self.param_1 = torch.nn.Parameter(
            torch.empty(self.hidden_dim, self.max_order, self.hidden_dim)
        )

        self.param_2 = torch.nn.Parameter(
            torch.empty(self.out_dim, self.max_order, self.hidden_dim)
        )

        if self.bias:
            self.bias_1 = torch.nn.Parameter(
                torch.empty(self.hidden_dim, self.max_order)
            )
            self.bias_2 = torch.nn.Parameter(torch.empty(self.out_dim, self.max_order))

        self.lin_skip = torch.nn.Linear(self.in_dim, self.out_dim, bias=self.bias)

        self.layer_norm = LayerNorm(self.in_dim)
        self.dropout = torch.nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the neural network module.

        Initializes the parameters of the module using the Kaiming uniform initialization for the `param_1` and `param_2` tensors.
        If the `bias` flag is set to True, initializes the biases (`bias_1` and `bias_2`) using the uniform initialization within a specific range.

        Returns:
            None
        """

        torch.nn.init.kaiming_uniform_(self.param_1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.param_2, a=math.sqrt(5))

        if self.bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.param_1)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias_1, -bound, bound)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.param_2)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias_2, -bound, bound)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_embedding: Tensor,
        lframes: LFrames,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the TensorMACE layer.

        Args:
            x (Tensor): Input node features of shape (num_nodes, input_dim).
            edge_index (Tensor): Graph edge indices of shape (2, num_edges).
            edge_embedding (Tensor): Edge embeddings of shape (num_edges, edge_dim).
            lframes (LFrames): LFrames object containing the local frames for each node.
            batch (Tensor, optional): Batch tensor of shape (num_nodes,). Defaults to None.

        Returns:
            Tensor: Output node features of shape (num_nodes, output_dim).
        """
        skip = x

        x = self.layer_norm(x, batch)

        # calculate the As
        A = self.propagate(
            edge_index, x=x, edge_embedding=edge_embedding, lframes=lframes
        )

        # calculate the Bs
        # Shape param_1: (hidden_dim, order, hidden_dim)
        # Shape A: (num_nodes, hidden_dim)
        tmp = torch.einsum("ih, onh -> ion", A, self.param_1)
        if self.bias:
            tmp = tmp + self.bias_1

        # TODO: Why is for loop faster than cumprod?
        B = torch.zeros(
            (x.shape[0], self.hidden_dim, self.max_order),
            device=x.device,
            dtype=x.dtype,
        )
        for i in range(self.max_order):
            B[:, :, i] = torch.prod(tmp[:, :, : i + 1], dim=-1)

        # B = torch.cumprod(tmp, dim=-1)

        # calculate the new node features
        # Shape param_2: (out_dim, order, hidden_dim)
        # Shape B: (num_nodes, hidden_dim, order)
        x = torch.einsum("ihn, onh -> ion", B, self.param_2)

        if self.bias:
            x = x + self.bias_2

        x = x.sum(dim=-1)

        return self.dropout(x) + self.lin_skip(skip)

    def message(self, x_j: Tensor, edge_embedding: Tensor) -> Tensor:
        """This method performs a message passing operation.

        Args:
            x_j (Tensor): The input tensor for node j.
            edge_embedding (Tensor): The edge embedding tensor.

        Returns:
            Tensor: The result of the message passing operation.
        """
        return self.edge_lin(x_j, edge_embedding)
