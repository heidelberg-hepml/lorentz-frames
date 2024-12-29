import torch


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
