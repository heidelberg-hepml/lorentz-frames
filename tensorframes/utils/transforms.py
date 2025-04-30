import torch
from typing import List

from tensorframes.utils.lorentz import lorentz_eye
from tensorframes.utils.restframe import restframe_boost


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
    std_eta: float = 0.1,
    n_max_std_eta: float = 3.0,
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
        n_max_std_eta: float
            Allowed number of standard deviations;
            used to sample from a truncated Gaussian
        device: str
        dtype: torch.dtype
        generator: torch.Generator

    Returns:
        final_trafo: torch.tensor of shape (*shape, 4, 4)
    """
    assert std_eta > 0
    ones = torch.ones(shape, device=device, dtype=torch.long)
    axis = torch.stack([0 * ones, 1 * ones], dim=0)
    angle = sample_rapidity(
        shape,
        std_eta=std_eta,
        n_max_std_eta=n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    boost = transform([axis], [angle])

    rotation = rand_rotation_uniform(shape, device, dtype, generator=generator)
    trafo = torch.einsum("...ij,...jk,...kl->...il", boost, rotation, boost)
    return trafo


def rand_general_lorentz(
    shape: List[int],
    std_eta: float = 0.1,
    n_max_std_eta: float = 3.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """
    Create N general Lorentz transformations as L=R*B,
    where R is a random uniform rotation in 3D and B
    is a random general boost transformation.

    Args:
        shape: List[int]
            Shape of the transformation matrices
        std_eta: float
            Standard deviation of rapidity
        n_max_std_eta: float
            Allowed number of standard deviations;
            used to sample from a truncated Gaussian
        device: str
        dtype: torch.dtype
        generator: torch.Generator

    Returns:
        final_trafo: torch.tensor of shape (*shape, 4, 4)
    """
    assert std_eta > 0
    boost = rand_general_boost(
        shape,
        std_eta,
        n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    rotation = rand_rotation_uniform(shape, device, dtype, generator=generator)

    trafo = torch.einsum("...ij,...jk->...ik", rotation, boost)
    return trafo


def rand_rotation_naive(
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

    Also, note that these rotations are not uniform. For uniform rotations,
    use rand_rotation_uniform (which is probably more powerful).

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
    This function is a special case of rand_rotation_naive

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
    std_eta: float = 0.1,
    n_max_std_eta: float = 3.0,
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
        n_max_std_eta: float
            Allowed number of standard deviations;
            used to sample from a truncated Gaussian
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
    angle2 = sample_rapidity(
        shape,
        std_eta=std_eta,
        n_max_std_eta=n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    return transform([axis1, axis2], [angle1, angle2])


def rand_rotation_uniform(
    shape: List[int],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """
    Create N rotation matrices embedded in the Lorentz group
    using quaternions. In contrast to the rand_rotation_naive
    function above, this is formally sound uniform sampling.

    Args:
        shape: List[int]
            Shape of the transformation matrices
        device: str
        dtype: torch.dtype
        generator: torch.Generator

    Returns:
        final_trafo: torch.tensor of shape (*shape, 4, 4)
    """
    # generate random quaternions
    u = torch.rand(*shape, 3, device=device, dtype=dtype, generator=generator)
    q1 = torch.sqrt(1 - u[..., 0]) * torch.sin(2 * torch.pi * u[..., 1])
    q2 = torch.sqrt(1 - u[..., 0]) * torch.cos(2 * torch.pi * u[..., 1])
    q3 = torch.sqrt(u[..., 0]) * torch.sin(2 * torch.pi * u[..., 2])
    q0 = torch.sqrt(u[..., 0]) * torch.cos(2 * torch.pi * u[..., 2])

    # create rotation matrix from quaternions
    R1 = torch.stack(
        [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        dim=-1,
    )
    R2 = torch.stack(
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
        dim=-1,
    )
    R3 = torch.stack(
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)],
        dim=-1,
    )
    R = torch.stack([R1, R2, R3], dim=-2)

    trafo = torch.eye(4, device=device, dtype=dtype).expand(*shape, 4, 4).clone()
    trafo[..., 1:, 1:] = R
    return trafo


def rand_general_boost(
    shape: List[int],
    std_eta: float = 0.1,
    n_max_std_eta: float = 3.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """
    Create N general Lorentz boosts from
    a vector of beta factors.

    Args:
        shape: List[int]
            Shape of the transformation matrices
        std_eta: float
            Standard deviation of rapidity
        n_max_std_eta: float
            Allowed number of standard deviations;
            used to sample from a truncated Gaussian
        device: str
        dtype: torch.dtype
        generator: torch.Generator

    Returns:
        final_trafo: torch.tensor of shape (*shape, 4, 4)
    """
    shape = shape + (3,)
    beta = sample_rapidity(
        shape,
        std_eta,
        n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    ones = torch.ones((*beta.shape[:-1], 1))
    beta = torch.cat([ones, beta], axis=-1)

    boost = restframe_boost(beta, is_beta=True)
    return boost


def sample_rapidity(
    shape,
    std_eta,
    n_max_std_eta=3.0,
    device="cpu",
    dtype=torch.float32,
    generator=None,
):
    angle = (
        torch.randn(*shape, device=device, dtype=dtype, generator=generator) * std_eta
    )
    truncate_mask = torch.abs(angle) > std_eta * n_max_std_eta
    while truncate_mask.any():
        new_angle = (
            torch.randn(*shape, device=device, dtype=dtype, generator=generator)
            * std_eta
        )
        angle[truncate_mask] = new_angle[truncate_mask]
        truncate_mask = torch.abs(angle) > std_eta * n_max_std_eta
    return angle
