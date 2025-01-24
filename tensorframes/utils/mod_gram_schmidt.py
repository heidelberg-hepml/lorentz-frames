from itertools import pairwise
import torch

from tensorframes.utils.lorentz import (
    lorentz_inner,
    lorentz_squarednorm,
)
from tensorframes.utils.orthogonalize import lorentz_cross
from tensorframes.utils.hep import get_deltaR


def mod_gram_schmidt(
    vecs: torch.tensor,
    eps: float = 1.0e-6,
) -> torch.tensor:
    """
    Applies the numerically stable Gram-Schmidt
    Args:
        vecs: torch.tensor of shape (3, N, 4) or (4, N, 4).
            If (3, N, 4) the last vector is calculated from the cross product.
        eps: nuerical regularization for the normalization of the vectors.
    """
    assert vecs.shape[0] == 3 or vecs.shape[0] == 4

    n_vectors = len(vecs)

    def normalize(v):
        norm = lorentz_squarednorm(v).unsqueeze(-1)
        norm = norm.abs().sqrt()  # could also multiply by torch.sign(norm)
        return v / (norm + eps)

    v_nexts = vecs.clone()
    orthogonal_vecs = [vecs[0]]
    for i in range(1, n_vectors):
        for k in range(i, n_vectors):
            v_inner = lorentz_inner(v_nexts[k], orthogonal_vecs[i - 1]).unsqueeze(-1)
            v_norm = lorentz_squarednorm(orthogonal_vecs[i - 1]).unsqueeze(-1)
            v_nexts[k] = v_nexts[k] - orthogonal_vecs[i - 1] * v_inner / v_norm
        orthogonal_vecs.append(v_nexts[i])
    if n_vectors == 3:
        last_vec = lorentz_cross(*orthogonal_vecs)
        orthogonal_vecs.append(last_vec)

    orthogonal_vecs = [normalize(v) for v in orthogonal_vecs]
    orthogonal_vecs = torch.stack(orthogonal_vecs)
    return orthogonal_vecs
