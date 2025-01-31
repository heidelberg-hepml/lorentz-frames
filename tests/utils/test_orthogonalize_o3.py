import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS

from tensorframes.utils.orthogonalize_o3 import orthogonalize_cross_o3


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_orthogonalize_o3(batch_dims):
    v1 = torch.randn(batch_dims + [3])
    v2 = torch.randn(batch_dims + [3])

    orthogonal_vecs = orthogonalize_cross_o3([v1, v2])

    for i1, v1 in enumerate(orthogonal_vecs):
        for i2, v2 in enumerate(orthogonal_vecs):
            inner = (v1 * v2).sum(dim=-1)
            target = torch.ones_like(inner) if i1 == i2 else torch.zeros_like(inner)
            torch.testing.assert_close(inner, target, **TOLERANCES)
