from itertools import pairwise
import torch

from tensorframes.utils.lorentz import (
    lorentz_inner,
    lorentz_squarednorm,
    lorentz_metric,
)

from tensorframes.utils.hep import get_deltaR


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


def orthogonalize_cross(vecs, eps=1e-10):
    n_vectors = len(vecs)
    assert n_vectors == 3

    vecs = [normalize(v, eps) for v in vecs]

    # orthogonalize vectors with repeated cross products
    orthogonal_vecs = [vecs[0]]
    for i in range(1, n_vectors + 1):
        v_next = lorentz_cross(*orthogonal_vecs, *vecs[i:])
        assert torch.isfinite(v_next).all()
        orthogonal_vecs.append(normalize(v_next, eps))

    return orthogonal_vecs


def orthogonalize_gramschmidt(
    vecs: torch.tensor,
    eps: float = 1e-10,
) -> torch.tensor:
    """
    Applies the numerically stable Gram-Schmidt
    Args:
        vecs: torch.tensor of shape (3, N, 4) or (4, N, 4).
            If (3, N, 4) the last vector is calculated from the cross product.
        eps: nuerical regularization for the normalization of the vectors.
    """
    n_vectors = len(vecs)
    assert n_vectors == 3 or n_vectors == 4

    vecs = [normalize(v, eps) for v in vecs]

    v_nexts = [v for v in vecs]
    orthogonal_vecs = [vecs[0]]
    for i in range(1, n_vectors):
        for k in range(i, n_vectors):
            v_inner = lorentz_inner(v_nexts[k], orthogonal_vecs[i - 1]).unsqueeze(-1)
            v_norm = lorentz_squarednorm(orthogonal_vecs[i - 1]).unsqueeze(-1)
            v_nexts[k] = v_nexts[k] - orthogonal_vecs[i - 1] * v_inner / (v_norm + eps)
        orthogonal_vecs.append(normalize(v_nexts[i], eps))
    if n_vectors == 3:
        last_vec = normalize(lorentz_cross(*orthogonal_vecs), eps)
        orthogonal_vecs.append(last_vec)

    orthogonal_vecs = [v for v in orthogonal_vecs]
    return orthogonal_vecs


def regularize_lightlike(
    vecs: torch.tensor,
    exception_eps: float = 1e-8,
    sample_eps: float = 1.0e-7,
    rejection_regularize=False,
):
    """
    Regularize the inputs to avoid lightlike vectors
    Args:
        vecs (Tensor): torch tensor of shape (3, N, 4) with N the batch dimension
        exception_eps (float): threshold applied to the criterion
        sample_eps (float): rescaling applied to the sampled four vectors
    Returns:
        tensor: regularized four vectors
    """
    if rejection_regularize:
        inners = torch.stack([lorentz_inner(v, v) for v in vecs])
        mask = (inners.abs() < exception_eps).to(vecs.device)[:, 0]
        return vecs[torch.argsort(mask)]

    assert vecs.shape[0] == 3

    inners = torch.stack([lorentz_inner(v, v) for v in vecs])
    sample = sample_eps * torch.randn(vecs.shape, dtype=vecs.dtype, device=vecs.device)
    mask = (inners.abs() < exception_eps)[..., None].expand_as(sample)
    vecs = vecs + sample * mask

    return vecs


def regularize_collinear(
    vecs: torch.tensor,
    exception_eps: float = 1e-6,
    sample_eps: float = 1.0e-5,
    rejection_regularize=False,
):
    """
    Regularize the inputs to avoid collinear vectors
    Args:
        vecs (Tensor): torch tensor of shape (3, N, 4) with N the batch dimension
        exception_eps (float): threshold applied to the criterion
        sample_eps (float): rescaling applied to the sampled four vectors
    Returns:
        tensor: regularized four vectors
    """
    if rejection_regularize:
        error = True
        safety = 10
        while error and (safety := safety - 1) > 0:
            error = False
            v_pairs = torch.cat((vecs[:3], vecs[0][None, ...]))
            deltaRs = torch.stack([get_deltaR(v, vp) for v, vp in pairwise(v_pairs)])
            mask = deltaRs < exception_eps
            if mask.sum() != 0:
                error = True
            mask = torch.cat(
                (
                    mask,
                    torch.full(
                        (vecs.shape[0] - 3, *mask.shape[1:]), True, device=vecs.device
                    ),
                ),
                dim=0,
            )

            indices = torch.argmax(mask.to(int), dim=0).tolist()
            temp = vecs[indices, torch.arange(vecs.shape[-2])]
            vecs[indices, torch.arange(vecs.shape[-2])] = vecs[3].clone()
            vecs[3:] = torch.cat((vecs[4:], temp.unsqueeze(0)))
        return vecs

    assert vecs.shape[0] == 3

    v_pairs = torch.cat((vecs, vecs[0][None, ...]))
    deltaRs = torch.stack([get_deltaR(v, vp) for v, vp in pairwise(v_pairs)])
    sample = sample_eps * torch.randn(vecs.shape, dtype=vecs.dtype, device=vecs.device)
    mask = (deltaRs < exception_eps)[..., None].expand_as(sample)
    vecs = vecs + sample * mask

    return vecs


def regularize_coplanar(
    vecs: torch.tensor,
    exception_eps: float = 1e-7,
    sample_eps: float = 1.0e-6,
    rejection_regularize=False,
):
    """
    Regularize the inputs to avoid collinear vectors
    Args:
        vecs (Tensor): torch tensor of shape (3, N, 4) with N the batch dimension
        exception_eps (float): threshold applied to the criterion
        sample_eps (float): rescaling applied to the sampled four vectors
    Returns:
        tensor: regularized four vectors
    """
    if rejection_regularize:
        error = True
        safety = 10
        while error and (safety := safety - 1) > 0:
            error = False
            cross_norm = lorentz_squarednorm(lorentz_cross(vecs[0], vecs[1], vecs[2]))
            mask = cross_norm.abs() < exception_eps
            vecs[2:, mask] = torch.cat(
                (vecs[3:, mask], vecs[2, mask].unsqueeze(0)), dim=0
            )

            if mask.sum() != 0:
                error = True
        return vecs

    assert vecs.shape[0] == 3

    cross_norm = lorentz_squarednorm(lorentz_cross(vecs[0], vecs[1], vecs[2]))
    sample = sample_eps * torch.randn(vecs.shape, dtype=vecs.dtype, device=vecs.device)
    mask = (cross_norm.abs() < exception_eps)[None, :, None].expand_as(sample)
    vecs = vecs + sample * mask

    return vecs


def order_vectors(
    trafo,
):
    # sort vectors by norm -> first vector has >0 norm
    # this is necessary to get valid Lorentz transforms
    # see paper for why at most one vector can have >0 norm
    vecs = [trafo[..., i, :] for i in range(4)]
    norm = torch.stack([lorentz_squarednorm(v) for v in vecs], dim=-1)
    pos_norm = norm > 0
    num_pos_norm = pos_norm.sum(dim=-1)
    assert (
        num_pos_norm == 1
    ).all(), f"Warning: find different number of norm>0 vectors: {torch.unique(num_pos_norm)}"
    old_trafo = trafo.clone()
    trafo[..., 0, :] = old_trafo[pos_norm]
    trafo[..., 1:, :] = old_trafo[~pos_norm].view(-1, 3, 4)

    return trafo


def cross_trafo(vecs, regularize=True, rejection_regularize=False, eps=1e-10):
    if regularize:
        vecs = regularize_collinear(vecs, rejection_regularize=rejection_regularize)
        vecs = regularize_coplanar(vecs, rejection_regularize=rejection_regularize)
        vecs = regularize_lightlike(vecs, rejection_regularize=rejection_regularize)
    vecs = vecs[:3]

    orthogonal_vecs = orthogonalize_cross(vecs, eps)
    trafo = torch.stack(orthogonal_vecs, dim=-2)

    # turn into transformation matrix
    metric = lorentz_metric(trafo.shape[:-2], device=trafo.device, dtype=trafo.dtype)
    trafo = trafo @ metric
    trafo = order_vectors(trafo)

    return trafo


def gramschmidt_trafo(vecs, regularize=True, rejection_regularize=False, eps=1e-10):
    if regularize:
        vecs = regularize_collinear(vecs, rejection_regularize=rejection_regularize)
        vecs = regularize_coplanar(vecs, rejection_regularize=rejection_regularize)
        vecs = regularize_lightlike(vecs, rejection_regularize=rejection_regularize)
    vecs = vecs[:3]

    orthogonal_vecs = orthogonalize_gramschmidt(vecs, eps)
    trafo = torch.stack(orthogonal_vecs, dim=-2)

    # turn into transformation matrix
    metric = lorentz_metric(trafo.shape[:-2], device=trafo.device, dtype=trafo.dtype)
    trafo = trafo @ metric
    trafo = order_vectors(trafo)

    return trafo


def normalize(v, eps=1e-10):
    norm = lorentz_squarednorm(v).unsqueeze(-1)
    norm = norm.abs().sqrt()  # could also multiply by torch.sign(norm)
    return v / (norm + eps)
