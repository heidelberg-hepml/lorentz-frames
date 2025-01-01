import torch

from tensorframes.utils.lorentz import lorentz_squarednorm, lorentz_eye, lorentz_metric


def reflect_single(v):
    """
    Create transformation matrix of a reflection based on a vector

    trafo = \delta^\mu_\nu - 2 * v^\mu v_nu / (v^\rho v_\rho)

    Args:
    v: torch.tensor of shape (*dims, 4)

    Returns:
    trafo: torch.tensor of shape (*dims, 4, 4)
    """
    assert v.shape[-1] == 4
    dims = v.shape[:-1]

    squarednorm = lorentz_squarednorm(v)[..., None, None]
    eye = lorentz_eye(dims, device=v.device, dtype=v.dtype)
    metric = lorentz_metric(dims, device=v.device, dtype=v.dtype)

    v_upper = v.unsqueeze(-1).repeat([1] * len(dims) + [1, 4])
    v_lower = torch.einsum("...ij,...j->...i", metric, v)
    v_lower = v_lower.unsqueeze(-2).repeat([1] * len(dims) + [4, 1])

    trafo = eye - 2 * v_upper * v_lower / squarednorm
    return trafo


def reflect_list(vs):
    """
    Build general transformation as a product of reflections
    Any Lorentz transformation can be written in this form
    because of the Cartan Dieudonne theorem

    Args:
    vs: List[torch.tensor] with list elements of shape (*dims, 4)

    Returns:
    trafo: torch.tensor of shape (*dims, 4, 4)
    """
    assert len(vs) > 0

    trafo = lorentz_eye(vs[0].shape[:-1], device=vs[0].device, dtype=vs[0].dtype)
    for v in vs:
        trafo_v = reflect_single(v)
        trafo = torch.einsum("...ij,...jk->...ik", trafo_v, trafo)
    return trafo
