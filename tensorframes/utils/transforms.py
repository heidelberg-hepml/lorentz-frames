import torch
from typing import List

from tensorframes.utils.lorentz import lorentz_eye


def get_trafo_type(axis):
    return torch.any(axis == 0, dim=0)


def transform(
    axes: List[int],
    angles: List[torch.Tensor],
    use_float64=True,
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
        use_float64: bool

    Returns:
        final_trafo: torch.Tensor of shape (*dims, 4, 4)
    """
    assert len(axes) == len(angles)
    dims = angles[0].shape
    assert all([angle.shape == dims for angle in angles])
    assert all([axis[0].shape == dims for axis in axes])
    assert all([axis[1].shape == dims for axis in axes])

    in_dtype = angles[0].dtype
    dtype = torch.float64 if use_float64 else in_dtype
    angles = [a.to(dtype=dtype) for a in angles]

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
    return final_trafo.to(in_dtype)


def rand_lorentz(
    shape: List[int],
    std_eta: float = 0.5,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """
    Create N transformation matrices

    We create them as boost * rotation * boost
    The last boost is necessary to get a general transformation,
    same story as for the rotations.

    Args:
        shape: List[int]
            Shape of the transformation matrices
        std_eta: float
            Standard deviation of rapidity
        device: str
        dtype: torch.dtype
        generator: torch.Generator

    Returns:
        final_trafo: torch.tensor of shape (*shape, 4, 4)
    """
    assert std_eta > 0
    ones = torch.ones(shape, device=device, dtype=torch.long)
    axis = torch.stack([0 * ones, 1 * ones], dim=0)
    angle = (
        torch.randn(*shape, device=device, dtype=dtype, generator=generator) * std_eta
    )
    boost = transform([axis], [angle])

    rotation = rand_rotation(shape, device, dtype, generator=generator)
    trafo = torch.einsum("...ij,...jk,...kl->...il", boost, rotation, boost)
    return trafo


def rand_rotation(
    shape: List[int],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """
    Create N rotation matrices embedded in the Lorentz group
    This function is very similar to rand_lorentz,
    differing only in how axis and angle are created

    Note that 3 rotations are enough to cover the whole group (Euler angles)
    (two rotations are only enough for point masses)

    Args:
        shape: List[int]
            Shape of the transformation matrices
        device: str
        dtype: torch.dtype
        generator: torch.Generator

    Returns:
        final_trafo: torch.tensor of shape (*shape, 4, 4)
    """
    ones = torch.ones(shape, device=device, dtype=torch.long)
    axes = [
        torch.stack([ones, 2 * ones], dim=0),
        torch.stack([ones, 3 * ones], dim=0),
        torch.stack([2 * ones, 3 * ones], dim=0),
    ]

    angles = []
    for _ in range(len(axes)):
        angle = (
            torch.rand(*shape, device=device, dtype=dtype, generator=generator)
            * 2
            * torch.pi
        )
        angles.append(angle)

    return transform(axes, angles)


def rand_xyrotation(
    shape: List[int],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """
    Create N xy-plane rotation matrices embedded in the Lorentz group
    This function is a special case of rand_rotation

    Args:
        shape: List[int]
            Shape of the transformation matrices
        device: str
        dtype: torch.dtype
        generator: torch.Generator

    Returns:
        final_trafo: torch.tensor of shape (*shape, 4, 4)
    """
    axis = torch.tensor([1, 2], dtype=torch.long, device=device)
    axis = axis.view(2, *([1] * len(shape))).repeat(1, *shape)
    angle = (
        torch.rand(*shape, device=device, dtype=dtype, generator=generator)
        * 2
        * torch.pi
    )
    return transform([axis], [angle])


def rand_ztransform(
    shape: List[int],
    std_eta: float = 0.5,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """
    Create N combinations of rotations around and boosts along the z-axis.
    This transformation is common in LHC physics.

    Args:
        shape: List[int]
            Shape of the transformation matrices
        std_eta: float
            Standard deviation of rapidity
        device: str
        dtype: torch.dtype
        generator: torch.Generator

    Returns:
        final_trafo: torch.tensor of shape (*shape, 4, 4)
    """
    # rotation around z-axis
    axis1 = torch.tensor([1, 2], dtype=torch.long, device=device)
    axis1 = axis1.view(2, *([1] * len(shape))).repeat(1, *shape)
    angle1 = (
        torch.rand(*shape, device=device, dtype=dtype, generator=generator)
        * 2
        * torch.pi
    )

    # boost along z-axis
    axis2 = torch.tensor([0, 3], dtype=torch.long, device=device)
    axis2 = axis2.view(2, *([1] * len(shape))).repeat(1, *shape)
    angle2 = (
        torch.rand(*shape, device=device, dtype=dtype, generator=generator) * std_eta
    )

    return transform([axis1, axis2], [angle1, angle2])
