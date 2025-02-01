import torch


def orthogonalize_o3(vecs, method="cross", eps_norm=1e-10, eps_reg=1e-4):
    """
    Orthogonalization tools for O(3) vectors

    Args:
        vecs: List of torch.tensor of shape (*dims, 3)
            Vectors to be orthogonalized
        method: str
            Method for orthogonalization. Options are "cross" and "gramschmidt".
        eps_norm: float
            Numerical regularization for the normalization of the vectors.
        eps_reg: float
            Controls when collinear vectors are regularized.
    """
    # regularization -> catch collinear vectors
    diff_norm = torch.linalg.norm(vecs[0] - vecs[1], dim=-1)
    mask = diff_norm < eps_reg
    vecs[1][mask] += eps_reg * torch.randn_like(vecs[1][mask])

    # orthonormalization
    if method == "cross":
        return orthogonalize_cross_o3(vecs, eps_norm)
    elif method == "gramschmidt":
        return orthogonalize_gramschmidt_o3(vecs, eps_norm)
    else:
        raise ValueError(f"Orthogonalization method {method} not implemented")


def orthogonalize_cross_o3(vecs, eps=1e-10):
    n_vectors = len(vecs)
    assert n_vectors == 2

    vecs = [normalize_o3(v, eps) for v in vecs]

    orthogonal_vecs = [vecs[0]]
    for i in range(1, n_vectors + 1):
        v_next = torch.cross(*orthogonal_vecs, *vecs[i:], dim=-1)
        assert torch.isfinite(v_next).all()
        orthogonal_vecs.append(normalize_o3(v_next, eps))

    return orthogonal_vecs


def orthogonalize_gramschmidt_o3(vecs, eps=1e-10):
    n_vectors = len(vecs)
    assert n_vectors == 2

    vecs = [normalize_o3(v, eps) for v in vecs]

    v_nexts = [v for v in vecs]
    orthogonal_vecs = [vecs[0]]

    # gram schmidt procedure
    for i in range(1, n_vectors):
        for k in range(i, n_vectors):
            v_inner = torch.sum(
                v_nexts[k] * orthogonal_vecs[i - 1], dim=-1, keepdim=True
            )
            v_nexts[k] = v_nexts[k] + orthogonal_vecs[i - 1] * v_inner
        orthogonal_vecs.append(normalize_o3(v_nexts[i], eps))

    # last vector from cross product
    last_vec = torch.cross(*orthogonal_vecs, dim=-1)
    orthogonal_vecs.append(normalize_o3(last_vec, eps))

    return orthogonal_vecs


def normalize_o3(v, eps=1e-10):
    norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / (norm + eps)
