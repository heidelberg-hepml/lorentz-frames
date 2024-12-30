import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS
from tensorframes.utils.transforms import rand_transform


@pytest.mark.parametrize("shape", BATCH_DIMS)
@pytest.mark.parametrize("n_range", [[1, 1], [3, 5]])
@pytest.mark.parametrize("std_eta", [0.1, 1, 2])
def test_lorentz_cross(shape, n_range, std_eta):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64  # some tests require higher precision

    # collect N different kinds of transformations
    transform = rand_transform(
        shape, n_range=n_range, std_eta=std_eta, device=device, dtype=dtype
    )
    assert torch.isfinite(transform).all()
    assert transform.shape == (*shape, 4, 4)

    # test that the transformation matrix T is orthogonal
    # i.e. T^T * M * T = M with the metric M = diag(1, -1, -1, -1)
    metric = torch.diag(
        torch.tensor([1, -1, -1, -1], device=transform.device, dtype=transform.dtype)
    )
    metric = metric.view((1,) * len(shape) + metric.shape).repeat(*shape, 1, 1)
    test = torch.einsum(
        "...ij,...jk,...kl->...il", transform, metric, transform.transpose(-1, -2)
    )
    torch.testing.assert_close(test, metric, **TOLERANCES)
