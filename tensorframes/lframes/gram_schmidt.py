import warnings
from typing import Union

import torch
from torch import Tensor


def gram_schmidt(
    x_axis: Tensor,
    y_axis: Tensor,
    z_axis: Union[Tensor, None] = None,
    eps: float = 1e-6,
    normalized: bool = True,
    exceptional_choice: str = "random",
) -> Tensor:
    """Applies the Gram-Schmidt process to a set of input vectors to orthogonalize them.

    Args:
        x_axis (Tensor): The first input vector. shape (N, 3)
        y_axis (Tensor): The second input vector. shape (N, 3)
        z_axis (Tensor, optional): The third input vector. shape (N, 3) Defaults to None. If None, the third vector is computed as the cross product of the first two.
        eps (float, optional): A small value used for numerical stability. Defaults to 1e-6.
        normalized (bool, optional): Whether to normalize the output vectors. Defaults to True.
        exceptional_choice (str, optional): The method to handle exceptional cases where the input vectors have zero length.
            Can be either "random" to use a random vector instead, or "zero" to set the vectors to zero.
            Defaults to "random".

    Returns:
        Tensor: A tensor containing the orthogonalized vectors the tensor has shape (N, 3, 3).

    Raises:
        ValueError: If the exceptional_choice parameter is not recognized.
        AssertionError: If z_axis has zero length.
    """
    x_length = torch.linalg.norm(x_axis, dim=-1, keepdim=True)

    x_zero_mask = (x_length < eps).squeeze()
    if torch.any(x_zero_mask):
        warnings.warn("x_axis has zero length")
        # print("x_axis has zero length", x_zero_mask.sum())
        if exceptional_choice == "random":
            x_axis = torch.where(
                x_zero_mask[:, None], x_axis + torch.rand_like(x_axis), x_axis
            )
            x_length = torch.linalg.norm(x_axis, dim=-1, keepdim=True)
        elif exceptional_choice == "zero":
            x_axis = torch.where(x_zero_mask[:, None], torch.zeros_like(x_axis), x_axis)
            x_length = torch.where(x_zero_mask[:, None], eps, x_length)
        else:
            raise ValueError(f"exceptional_choice {exceptional_choice} not recognized")

    if normalized:
        x_axis = x_axis / torch.clamp(x_length, eps)
        y_axis = y_axis - torch.einsum("ij,ij->i", y_axis, x_axis)[:, None] * x_axis
    else:
        y_axis = y_axis - torch.einsum("ij,ij->i", y_axis, x_axis)[
            :, None
        ] * x_axis / torch.clamp(torch.square(x_length), eps)

    # handle the case where y_axis is zero (this can happen if x and y are parallel)
    y_length = torch.linalg.norm(y_axis, dim=-1, keepdim=True)

    y_zero_mask = (y_length < eps).squeeze()
    if torch.any(y_zero_mask):
        # print("y_axis has zero length", y_zero_mask.sum())
        if exceptional_choice == "random":
            y_axis = torch.where(
                y_zero_mask[:, None], y_axis + torch.rand_like(y_axis), y_axis
            )
            y_axis = torch.where(
                y_zero_mask[:, None],
                y_axis - torch.einsum("ij,ij->i", y_axis, x_axis)[:, None] * x_axis,
                y_axis,
            )
            y_length = torch.linalg.norm(y_axis, dim=-1, keepdim=True)
        elif exceptional_choice == "zero":
            y_axis = torch.where(y_zero_mask[:, None], torch.zeros_like(y_axis), y_axis)
            y_length = torch.where(y_zero_mask[:, None], eps, y_length)
        else:
            raise ValueError(f"exceptional_choice {exceptional_choice} not recognized")

    if normalized:
        y_axis = y_axis / torch.clamp(y_length, eps)

    if z_axis is None:
        lframes = torch.stack(
            [x_axis, y_axis, torch.linalg.cross(x_axis, y_axis, dim=-1)], dim=-2
        )
    else:
        if normalized:
            z_tmp = torch.linalg.cross(x_axis, y_axis, dim=-1)
            z_dot = torch.einsum("ij, ij -> i", z_tmp, z_axis)
            z_dot_mask = torch.sign(z_dot).abs() < 0.5
            z_sign = torch.where(
                z_dot_mask,
                torch.sign(torch.rand_like(z_dot_mask.float()) - 0.5),
                torch.sign(z_dot),
            )
            z_axis = z_tmp * z_sign[:, None]
        else:
            z_axis = (
                z_axis
                - torch.einsum("ij,ij->i", z_axis, x_axis)[:, None]
                * x_axis
                / torch.clamp(torch.square(x_length), eps)
                - torch.einsum("ij,ij->i", z_axis, y_axis)[:, None]
                * y_axis
                / torch.clamp(torch.square(y_length), eps)
            )

        # handle the case where y_axis is zero (this can happen if x and y are parallel)
        z_length = torch.linalg.norm(z_axis, dim=-1, keepdim=True)

        if normalized:
            z_axis = z_axis / torch.clamp(z_length, eps)
            z_length = torch.linalg.norm(z_axis, dim=-1, keepdim=True)

        z_zero_mask = (z_length < eps).squeeze()

        if torch.any(z_zero_mask):
            if exceptional_choice == "random":
                flip_vec = torch.sign(torch.rand(z_zero_mask.sum()) - 0.5).to(
                    device=x_axis.device
                )

                crossvec = torch.linalg.cross(
                    x_axis[z_zero_mask], y_axis[z_zero_mask], dim=-1
                )

                flipped_vec = torch.einsum("i,ij->ij", flip_vec, crossvec)

                z_axis[z_zero_mask] = flipped_vec

                z_length = torch.linalg.norm(z_axis, dim=-1, keepdim=True)
            elif exceptional_choice == "zero":
                z_axis = torch.where(
                    z_zero_mask[:, None], torch.zeros_like(z_axis), z_axis
                )
                z_length = torch.where(z_zero_mask[:, None], eps, z_length)
            else:
                raise ValueError(
                    f"exceptional_choice {exceptional_choice} not recognized"
                )

        lframes = torch.stack([x_axis, y_axis, z_axis], dim=-2)

    return lframes
