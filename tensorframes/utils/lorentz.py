import torch


def lorentz_inner(v1, v2):
    """Lorentz inner product, i.e v1*g*v2"""
    prod = v1 * v2
    prod[..., 1:] *= -1
    return prod.sum(dim=-1)


def lorentz_squarednorm(v):
    """Lorentz norm, i.e. v*g*v"""
    return lorentz_inner(v, v)


def lorentz_eye(dims, device=torch.device("cpu"), dtype=torch.float32):
    """
    Create a identity matrix of shape (*dims, 4, 4)
    """
    eye = torch.eye(4, dtype=dtype, device=device)
    eye = eye.view((1,) * len(dims) + eye.shape).repeat(*dims, 1, 1)
    return eye


def lorentz_metric(dims, device=torch.device("cpu"), dtype=torch.float32):
    """
    Create a metric tensor of shape (*dims, 4, 4)
    """
    eye = torch.eye(4, device=device, dtype=dtype)
    eye[1:, 1:] *= -1
    eye = eye.view((1,) * len(dims) + eye.shape).repeat(*dims, 1, 1)
    return eye


def lorentz_cross(v1, v2, v3):
    """
    Compute the cross product in Minkowski space.

    Args:
        v1, v2, v3: Tensors of shape (*dims, 4)

    Returns:
        v4: Tensor of shape (*dims, )
    """
    assert v1.shape[-1] == 4
    assert v1.shape == v2.shape and v1.shape == v3.shape

    mat = torch.stack([v1, v2, v3], dim=-1)

    # euclidean fully antisymmetric product
    v4 = []
    for n in range(4):
        minor = torch.cat([mat[..., :n, :], mat[..., n + 1 :, :]], dim=-2)
        contribution = (-1) ** n * torch.det(minor)
        v4.append(contribution)
    v4 = torch.stack(v4, dim=-1)

    # raise indices with metric tensor
    v4 *= torch.tensor([1.0, -1.0, -1.0, -1.0], device=v1.device, dtype=v1.dtype)
    return v4
