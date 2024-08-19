from typing import Callable, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import MLP as GeometricMLP
from torch_geometric.typing import NoneType
from torchvision.ops import MLP as TorchMLP

from tensorframes.reps.irreps import Irreps
from tensorframes.reps.tensorreps import TensorReps


class MLPWrapped(Module):
    """Multi-Layer Perceptron (MLP) module.

    Which can be used as the one from torchvision but also has the option to use torch_geometric
    MLPs.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dropout: float = 0.0,
        use_torchvision: bool = True,
        **kwargs
    ):
        """Initializes the MLP module.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (List[int]): List of hidden layer sizes.
            norm_layer (Optional[Callable[..., torch.nn.Module]], optional): Normalization layer to use. Defaults to None.
            activation_layer (Optional[Callable[..., torch.nn.Module]], optional): Activation layer to use. Defaults to torch.nn.ReLU.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            use_torchvision (bool, optional): Whether to use torchvision MLP implementation. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

        self.use_torchvision = use_torchvision
        self.out_dim = hidden_channels[-1]

        if self.use_torchvision:
            self.mlp = TorchMLP(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                dropout=dropout,
                **kwargs
            )
        else:
            self.mlp = GeometricMLP(
                channel_list=[in_channels] + list(hidden_channels),
                dropout=dropout,
                norm=norm_layer,
                act=activation_layer,
                **kwargs
            )

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        return_emb: NoneType = None,
    ) -> Tensor:
        r"""Forward pass of the MLP module.

        Args:
            x (torch.Tensor): The source tensor.
            batch (torch.Tensor, optional): The batch vector :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example. Only needs to be passed in case the underlying normalization
                layers require the `batch` information. (default: `None`)
            batch_size (int, optional): The number of examples :math:`B`. Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization layers require the `batch` information.
                (default: `None`)
            return_emb (bool, optional): If set to `True`, will additionally return the embeddings before execution
                of the final output layer. (default: `False`)

        Returns:
            torch.Tensor: The output tensor.
        """
        if self.use_torchvision:
            return self.mlp(x)
        else:
            return self.mlp(x, batch=batch, batch_size=batch_size, return_emb=return_emb)


class MLP(torch.nn.Module):
    """An MLP module which uses reps for the input and output dimensions."""

    def __init__(
        self,
        in_reps: Union[TensorReps, Irreps],
        out_reps: Union[TensorReps, Irreps],
        hidden_layers,
        **mlp_kwargs
    ) -> None:
        """Initialize the MLP class.

        Args:
            in_reps (Union[TensorReps, Irreps]): The input representations.
            out_reps (Union[TensorReps, Irreps]): The output representations.
            hidden_layers (list): A list of integers representing the sizes of the hidden layers.
            **mlp_kwargs: Additional keyword arguments to be passed to the MLP constructor.
        """

        super().__init__()
        self.in_reps = in_reps
        self.out_reps = out_reps
        self.hidden_layers = hidden_layers.copy()
        self.hidden_layers.append(out_reps.dim)

        self.mlp = TorchMLP(in_reps.dim, self.hidden_layers, **mlp_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass through the MLP network.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the MLP network.
        """
        return self.mlp(x)
