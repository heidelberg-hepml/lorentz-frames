import torch
from tensorframes.utils.utils import stable_arctanh


def deltaR(v1, v2):
    y1 = stable_arctanh(
        v1[..., 3] / torch.sqrt(v1[..., 1] ** 2 + v1[..., 2] ** 2 + v1[..., 3] ** 2)
    )
    y2 = stable_arctanh(
        v2[..., 3] / torch.sqrt(v2[..., 1] ** 2 + v2[..., 2] ** 2 + v2[..., 3] ** 2)
    )

    phi1 = torch.arctan2(v1[..., 2], v1[..., 1])
    phi2 = torch.arctan2(v2[..., 2], v2[..., 1])

    delta_y = y1 - y2
    delta_phi = phi1 - phi2

    dphi1 = delta_phi > torch.pi
    delta_phi[dphi1] -= 2 * torch.pi

    dphi2 = delta_phi < -torch.pi
    delta_phi[dphi2] += 2 * torch.pi
    return torch.sqrt(delta_y**2 + delta_phi**2)
