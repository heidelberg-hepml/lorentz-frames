import warnings
from typing import Tuple, Union

import torch

from tensorframes.lframes import LFrames
from tensorframes.nn.mlp import MLPWrapped
from tensorframes.reps import Irreps, TensorReps
from tensorframes.utils.quaternions import quaternions_to_matrix


class QuaternionsUpdateLFrames(torch.nn.Module):
    """Module for updating LFrames using quaternions."""

    def __init__(
        self,
        in_reps: Union[TensorReps, Irreps],
        hidden_channels: list,
        init_zero_angle: bool = False,
        eps: float = 1e-6,
        **mlp_kwargs
    ):
        """Initialize the UpdatingLFrames module.

        Args:
            in_reps (list): List of input representations.
            hidden_channels (list): List of hidden channel sizes for the MLP.
            init_zero_angle (bool, optional): Whether to initialize angle weights to zero. Defaults to False.
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
            **mlp_kwargs: Additional keyword arguments for the MLPWrapped module.
        """
        super().__init__()
        self.in_reps = in_reps
        self.eps = eps

        self.mlp = MLPWrapped(
            in_channels=self.in_reps.dim,
            hidden_channels=hidden_channels + [5],
            **mlp_kwargs,
        )
        self.coeffs_transform = self.in_reps.get_transform_class()

        if init_zero_angle:
            warnings.warn(
                "Make sure that the activation function is NOT ReLU, When using init_zero_angle = True."
            )
            self.set_angle_weights_to_zero()

    def set_angle_weights_to_zero(self):
        """Sets the relevant weights and biases to zero to achieve that the first output channel
        predicts zeros initially."""
        with torch.no_grad():
            if self.mlp.use_torchvision:
                # torchvision mlp
                self.mlp.mlp[-2].weight.data[1].zero_()
                self.mlp.mlp[-2].bias.data[1].zero_()
            else:
                # torch_geometric mlp
                self.mlp._modules["lins"][-1].weight.data[1].zero_()
                self.mlp._modules["lins"][-1].bias.data[1].zero_()

    def forward(
        self, x: torch.Tensor, lframes: LFrames, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, LFrames]:
        """Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor.
            lframes (LFrames): LFrames object.
            batch (torch.Tensor): Batch tensor.

        Returns:
            Tuple[torch.Tensor, LFrames]: Tuple containing the updated input tensor and LFrames object.
        """
        out = self.mlp(x, batch=batch)
        denominator = torch.where(out[..., 0].abs() < self.eps, self.eps, out[..., 0])
        angle = torch.arctan2(out[..., 1], denominator)
        axis = torch.nn.functional.normalize(out[..., 2:], p=2, dim=-1)
        rot_matr = quaternions_to_matrix(
            torch.cat(
                [torch.cos(angle / 2).unsqueeze(-1), torch.sin(angle / 2).unsqueeze(-1) * axis],
                dim=-1,
            )
        )

        new_lframes = LFrames(torch.einsum("ijk, ikn -> ijn", rot_matr, lframes.matrices))
        new_x = self.coeffs_transform(x, LFrames(rot_matr))

        return new_x, new_lframes
