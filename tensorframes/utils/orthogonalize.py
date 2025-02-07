import torch

from tensorframes.utils.lorentz import (
    lorentz_inner,
    lorentz_squarednorm,
    lorentz_metric,
    lorentz_cross,
)


def orthogonal_trafo(*args, **kwargs):
    """Put everything together to construct a valid Lorentz transformation"""
    out = orthogonalize(*args, **kwargs)
    if kwargs["return_frac"]:
        orthogonal_vecs, *frac = out
    else:
        orthogonal_vecs = out
    trafo = torch.stack(orthogonal_vecs, dim=-2)

    metric = lorentz_metric(trafo.shape[:-2], device=trafo.device, dtype=trafo.dtype)
    trafo = trafo @ metric
    trafo = timelike_first(trafo)
    return (trafo, *frac) if kwargs["return_frac"] else trafo


def orthogonalize(
    vecs,
    method="gramschmidt",
    eps_norm=1e-10,
    eps_reg_coplanar=1e-6,
    eps_reg_lightlike=1e-8,
    return_frac=False,
):
    """
    Wrapper for orthogonalization of O(1,3) vectors

    Args:
        vecs: List of torch.tensor of shape (*dims, 4)
            Vectors to be orthogonalized
        method: str
            Method for orthogonalization. Options are "cross" and "gramschmidt".
        eps_norm: float
            Numerical regularization for the normalization of the vectors.
        eps_reg_coplanar: float
            Controls when coplanar vectors are regularized.
        eps_reg_lightlike: float
            Controls when lightlike vectors are regularized.
        return_frac: bool

    Returns:
        orthogonal_vecs: List of torch.tensor of shape (*dims, 4)
            Orthogonalized vectors
    """
    assert len(vecs) == 3
    assert all(v.shape == vecs[0].shape for v in vecs)

    vecs, frac_lightlike = regularize_lightlike(vecs, eps_reg_lightlike)
    vecs, frac_coplanar = regularize_coplanar(vecs, eps_reg_coplanar)

    if method == "cross":
        trafo = orthogonalize_cross(vecs, eps_norm)
    elif method == "gramschmidt":
        trafo = orthogonalize_gramschmidt(vecs, eps_norm)
    else:
        raise ValueError(f"Orthogonalization method {method} not implemented")

    return (trafo, frac_lightlike, frac_coplanar) if return_frac else trafo


def orthogonalize_cross(vecs, eps_norm=1e-10):
    """Repeated cross products"""
    vecs = [normalize(v, eps_norm) for v in vecs]

    orthogonal_vecs = [vecs[0]]
    for i in range(1, len(vecs) + 1):
        v_next = lorentz_cross(*orthogonal_vecs, *vecs[i:])
        assert torch.isfinite(v_next).all()
        orthogonal_vecs.append(normalize(v_next, eps_norm))

    return orthogonal_vecs


def orthogonalize_gramschmidt(vecs, eps_norm=1e-10):
    """Gram-Schmidt orthogonalization algorithm"""
    vecs = [normalize(v, eps_norm) for v in vecs]

    v_nexts = [v for v in vecs]
    orthogonal_vecs = [vecs[0]]
    for i in range(1, len(vecs)):
        for k in range(i, len(vecs)):
            v_inner = lorentz_inner(v_nexts[k], orthogonal_vecs[i - 1]).unsqueeze(-1)
            v_norm = lorentz_squarednorm(orthogonal_vecs[i - 1]).unsqueeze(-1)
            v_nexts[k] = v_nexts[k] - orthogonal_vecs[i - 1] * v_inner / (
                v_norm + eps_norm
            )
        orthogonal_vecs.append(normalize(v_nexts[i], eps_norm))
    last_vec = normalize(lorentz_cross(*orthogonal_vecs), eps_norm)
    orthogonal_vecs.append(last_vec)

    orthogonal_vecs = [v for v in orthogonal_vecs]
    return orthogonal_vecs


def timelike_first(trafo):
    """
    Re-order vectors such that the (single) timelike vector comes first
    This is necessary to get valid Lorentz transformations
    """
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


def regularize_lightlike(vecs, eps_reg_lightlike=1e-8):
    """
    Regularize lightlike vectors:
    Add a bit of noise to every lightlike vector
    """
    vecs_reg = []
    masks = []
    for v in vecs:
        inners = lorentz_inner(v, v)
        mask = inners.abs() < eps_reg_lightlike
        v_reg = v + eps_reg_lightlike * torch.randn_like(v) * mask.unsqueeze(-1)
        masks.append(mask)
        vecs_reg.append(v_reg)

    frac_lightlike = torch.stack(masks).any(dim=-1).float().mean().item()
    return vecs_reg, frac_lightlike


def regularize_coplanar(vecs, eps_reg_coplanar=1e-6):
    """
    Regularize coplanar vectors:
    Add a bit of noise to every triplet of coplanar vectors
    """
    assert len(vecs) == 3
    cross_norm = lorentz_squarednorm(lorentz_cross(*vecs))
    mask = cross_norm.abs() < eps_reg_coplanar

    vecs_reg = []
    for v in vecs:
        v_reg = v + eps_reg_coplanar * torch.randn_like(v) * mask.unsqueeze(-1)
        vecs_reg.append(v_reg)

    frac_coplanar = mask.float().mean().item()
    return vecs, frac_coplanar


def normalize(v, eps=1e-10):
    norm = lorentz_squarednorm(v).unsqueeze(-1)
    norm = norm.abs().sqrt()  # could also multiply by torch.sign(norm)
    return v / (norm + eps)
