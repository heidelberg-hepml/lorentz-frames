import torch

from tensorframes.utils.utils import stable_arctanh
from tensorframes.utils.transforms import transform


def restframe_transform(fourmomenta):
    """
    Create rest frame transformation matrices for given fourmomenta
    as a combination of rotations and boosts

    Strategy:
    1) Rotate around z-axis to set p_y=0
    2) Rotate around the y-axis to set p_z=0
    3) Boost along the x-axis to set p_x=0

    This transformation probably does not do the job,
    because it is invariant under rotations and not equivariant
    -> motivation for v2 below

    Args:
        fourmomenta: torch.tensor of shape (*dims, 4)

    Returns:
        final_trafo: torch.tensor of shape (*dims, 4, 4)
    """
    axes = torch.tensor(
        [[1, 2], [1, 3], [0, 1]], device=fourmomenta.device, dtype=torch.long
    )
    axes = axes.view(*axes.shape, *([1] * len(fourmomenta.shape[:-1])))
    axes = axes.repeat(1, 1, *fourmomenta.shape[:-1])
    axes = [ax for ax in axes]

    fm = fourmomenta
    angles0 = -torch.arctan(fm[..., 2] / fm[..., 1])
    angles1 = -torch.arctan(
        fm[..., 3] / (fm[..., 1] * torch.cos(angles0) - fm[..., 2] * torch.sin(angles0))
    )
    angles2 = -stable_arctanh(torch.linalg.norm(fm[..., 1:], dim=-1) / fm[..., 0])
    angles2 *= torch.sign(fm[..., 1])
    angles = [angles0, angles1, angles2]

    return transform(axes, angles)


def restframe_transform_v2(fourmomenta):
    """
    Create rest frame transformation matrices for given fourmomenta
    as a single transformation using a textbook formula

    Args:
        fourmomenta: torch.tensor of shape (*dims, 4)

    Returns:
        final_trafo: torch.tensor of shape (*dims, 4, 4)
    """
    beta = fourmomenta[..., 1:] / fourmomenta[..., [0]]
    beta2 = (beta**2).sum(dim=-1, keepdim=True)
    gamma = 1 / (1 - beta2).sqrt()

    # prepare entries of the trafo
    boost = -gamma * beta
    eye = torch.eye(3, device=fourmomenta.device, dtype=fourmomenta.dtype)
    eye = eye.view(*(1,) * len(fourmomenta.shape[:-1]), 3, 3).repeat(
        *fourmomenta.shape[:-1], 1, 1
    )
    rot = eye + (
        (gamma[..., None] - 1) * beta[..., None] * beta[..., None, :] / beta2[..., None]
    )

    # put trafo together
    trafo = torch.empty(
        *fourmomenta.shape[:-1],
        4,
        4,
        device=fourmomenta.device,
        dtype=fourmomenta.dtype
    )
    trafo[..., 0, 0] = gamma[..., 0]
    trafo[..., 1:, 1:] = rot
    trafo[..., 0, 1:] = boost
    trafo[..., 1:, 0] = boost
    return trafo
