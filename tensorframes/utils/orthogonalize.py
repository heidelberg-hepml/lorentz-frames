import torch

from tensorframes.utils.lorentz import lorentz_squarednorm


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


def orthogonalize_cross(vecs, eps):
    n_vectors = len(vecs)

    def normalize(v):
        norm = lorentz_squarednorm(v).unsqueeze(-1)
        norm = torch.where(norm > 0, norm.sqrt(), -(-norm).sqrt())
        norm = torch.where(norm > 0, norm + eps, norm - eps)  # avoid division by zero
        return v / norm

    vecs = [normalize(v) for v in vecs]

    # orthogonalize vectors with repeated cross products
    orthogonal_vecs = [vecs[0]]
    for i in range(1, n_vectors + 1):
        v_next = lorentz_cross(*orthogonal_vecs, *vecs[i:])
        assert torch.isfinite(v_next).all()
        orthogonal_vecs.append(normalize(v_next))

    return orthogonal_vecs
