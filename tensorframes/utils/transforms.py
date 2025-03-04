import torch
import numpy as np
from random import randint
from typing import List

from tensorframes.utils.lorentz import lorentz_eye


def get_trafo_type(axis):
    return torch.any(axis == 0, dim=0)


def transform(
    axes: List[int],
    angles: List[torch.Tensor],
):
    """
    Recursively build transformation matrices based on given lists of axes and angles

    Args:
        axes: List[torch.Tensor] with elements of shape (2,*dims)
            Axes along which the transformations are performed
            Note that this object contains the trafo_types information,
            because 'trafo_type = 1 if 0 in angle else 0'
        angles: List[torch.tensor] with elements of shape (*dims,)
            Angles used for the transformations,
            can be either rotation angles or rapidities

    Returns:
        final_trafo: torch.Tensor of shape (*dims, 4, 4)
    """
    assert len(axes) == len(angles)
    dims = angles[0].shape
    assert all([angle.shape == dims for angle in angles])
    assert all([axis[0].shape == dims for axis in axes])
    assert all([axis[1].shape == dims for axis in axes])

    final_trafo = lorentz_eye(dims, angles[0].device, angles[0].dtype)
    for axis, angle in zip(axes, angles):
        trafo = lorentz_eye(dims, angle.device, angle.dtype)
        trafo_type = get_trafo_type(axis)

        meshgrid = torch.meshgrid(*[torch.arange(d) for d in dims], indexing="ij")
        trafo[(*meshgrid, axis[0], axis[0])] = torch.where(
            trafo_type, torch.cosh(angle), torch.cos(angle)
        )
        trafo[(*meshgrid, axis[0], axis[1])] = torch.where(
            trafo_type, torch.sinh(angle), -torch.sin(angle)
        )
        trafo[(*meshgrid, axis[1], axis[0])] = torch.where(
            trafo_type, torch.sinh(angle), torch.sin(angle)
        )
        trafo[(*meshgrid, axis[1], axis[1])] = torch.where(
            trafo_type, torch.cosh(angle), torch.cos(angle)
        )
        final_trafo = torch.einsum("...jk,...kl->...jl", trafo, final_trafo)
    return final_trafo


def rand_lorentz(
    shape: List[int],
    n_range: List[int] = [3, 5],
    std_eta: float = 0.5,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    Create N transformation matrices

    Args:
        shape: List[int]
            Shape of the transformation matrices
        n_range: List[int] = [3, 5]
            Range of number of transformations
            Warning: For too many transformations, the matrix might not be orthogonal
            because numerical errors add up
        std_eta: float
            Standard deviation of rapidity
        device: str
        dtype: torch.dtype

    Returns:
        final_trafo: torch.tensor of shape (*shape, 4, 4)
    """
    assert std_eta > 0

    n_transforms = randint(*n_range)
    assert n_transforms > 0

    axes, angles = [], []
    for _ in range(n_transforms):
        axis0 = torch.randint(0, 4, shape, device=device)
        axis1 = (axis0 + torch.randint(1, 4, shape, device=device)) % 4
        assert (axis0 != axis1).all()
        axis = torch.stack([axis0, axis1], dim=0)
        axes.append(axis)

        trafo_type = get_trafo_type(axis)
        angle = torch.where(
            trafo_type,
            torch.randn(*shape, device=device, dtype=dtype) * std_eta,
            torch.rand(*shape, device=device, dtype=dtype) * 2 * torch.pi,
        )
        angles.append(angle)

    return transform(axes, angles)


def rand_rotation(
    shape: List[int],
    n_range: List[int] = [3, 5],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    Create N rotation matrices embedded in the Lorentz group
    This function is very similar to rand_lorentz,
    differing only in how axis and angle are created

    Args:
        shape: List[int]
            Shape of the transformation matrices
        n_range: List[int] = [3, 5]
            Range of number of transformations
            Warning: For too many transformations, the matrix might not be orthogonal
            because numerical errors add up
        device: str
        dtype: torch.dtype

    Returns:
        final_trafo: torch.tensor of shape (*shape, 4, 4)
    """
    n_transforms = randint(*n_range)
    assert n_transforms > 0

    axes, angles = [], []
    for _ in range(n_transforms):
        axis0 = torch.randint(1, 4, shape, device=device)
        axis1 = 1 + (axis0 - 1 + torch.randint(1, 3, shape, device=device)) % 3
        assert (axis0 != axis1).all()
        axis = torch.stack([axis0, axis1], dim=0)
        assert (axis != 0).all()
        axes.append(axis)

        angle = torch.rand(*shape, device=device, dtype=dtype) * 2 * torch.pi
        angles.append(angle)

    return transform(axes, angles)


def rand_xyrotation(
    shape: List[int],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    Create N xy-plane rotation matrices embedded in the Lorentz group
    This function is a special case of rand_rotation

    Args:
        shape: List[int]
            Shape of the transformation matrices
        device: str
        dtype: torch.dtype

    Returns:
        final_trafo: torch.tensor of shape (*shape, 4, 4)
    """
    axis = torch.tensor([1, 2], dtype=torch.long, device=device)
    axis = axis.view(2, *([1] * len(shape))).repeat(1, *shape)
    angle = torch.rand(*shape, device=device, dtype=dtype) * 2 * torch.pi
    return transform([axis], [angle])


def rand_boost(
    shape: List[int],
    n_range: List[int] = [3, 5],
    std_eta: float = 0.5,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    Create N boost matrices embedded in the Lorentz group
    This function is very similar to rand_lorentz,
    differing only in how axis and angle are created

    Args:
        shape: List[int]
            Shape of the transformation matrices
        n_range: List[int] = [3, 5]
            Range of number of transformations
            Warning: For too many transformations, the matrix might not be orthogonal
            because numerical errors add up
        std_eta: float
            Standard deviation of rapidity
        device: str
        dtype: torch.dtype

    Returns:
        final_trafo: torch.tensor of shape (*shape, 4, 4)
    """
    assert std_eta > 0

    n_transforms = randint(*n_range)
    assert n_transforms > 0

    axes, angles = [], []
    for _ in range(n_transforms):
        axis0 = torch.zeros(shape, device=device, dtype=torch.long)
        axis1 = torch.randint(1, 4, shape, device=device)
        assert (axis0 != axis1).all()
        axis = torch.stack([axis0, axis1], dim=0)
        axes.append(axis)

        angle = torch.rand(*shape, device=device, dtype=dtype) * std_eta
        angles.append(angle)
    return transform(axes, angles)
