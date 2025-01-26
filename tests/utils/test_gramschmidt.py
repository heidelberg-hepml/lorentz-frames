import torch
import pytest
from tests.constants import TOLERANCES

from tensorframes.utils.lorentz import (
    lorentz_inner,
)
from tensorframes.utils.gram_schmidt import gramschmidt_orthogonalize


@pytest.mark.parametrize("batch_dims", [[1000]])
@pytest.mark.parametrize("n_vectors", [3, 4])
def test_gram_schmidt(batch_dims, n_vectors):
    # check orthogonality after using the function
    dtype = torch.float64

    v1 = torch.randn(batch_dims + [4], dtype=dtype)
    v2 = torch.randn(batch_dims + [4], dtype=dtype)
    v3 = torch.randn(batch_dims + [4], dtype=dtype)
    if n_vectors == 4:
        v4 = torch.randn(batch_dims + [4], dtype=dtype)
        vecs = torch.stack([v1, v2, v3, v4])
    else:
        vecs = torch.stack([v1, v2, v3])

    orthogonal_vecs = gramschmidt_orthogonalize(vecs)

    for i1, v1 in enumerate(orthogonal_vecs):
        for i2, v2 in enumerate(orthogonal_vecs):
            inner = lorentz_inner(v1, v2)
            target = torch.ones_like(inner) if i1 == i2 else torch.zeros_like(inner)
            torch.testing.assert_close(inner.abs(), target, **TOLERANCES)
