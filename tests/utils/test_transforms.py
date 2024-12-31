import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS
from tensorframes.utils.transforms import (
    rand_transform,
    rand_rotation,
    rand_phirotation,
    restframe_transform,
)
from tensorframes.utils.lorentz import lorentz_norm


@pytest.mark.parametrize("shape", BATCH_DIMS)
@pytest.mark.parametrize("n_range", [[1, 1], [3, 5]])
@pytest.mark.parametrize("std_eta", [0.1, 1, 2])
@pytest.mark.parametrize(
    "transform_type", [rand_transform, rand_rotation, rand_phirotation]
)
def test_rand(shape, n_range, std_eta, transform_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64  # some tests require higher precision

    # collect N different kinds of transformations
    kwargs = {
        "shape": shape,
        "device": device,
        "dtype": dtype,
    }
    if transform_type in [rand_transform, rand_rotation]:
        kwargs["n_range"] = n_range
    if transform_type in [rand_transform]:
        kwargs["std_eta"] = std_eta
    transform = transform_type(**kwargs)
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


@pytest.mark.parametrize("shape", BATCH_DIMS)
@pytest.mark.parametrize("logm2_std", [0.1, 1, 2])
@pytest.mark.parametrize("logm2_mean", [-3, 0, 3])
def test_restframe_transform(shape, logm2_std, logm2_mean):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # sample Lorentz vectors
    logm2 = torch.randn(*shape, 1, device=device, dtype=dtype) * logm2_std + logm2_mean
    p3 = torch.randn(*shape, 3, device=device, dtype=dtype)
    E = torch.sqrt(logm2.exp() + (p3**2).sum(dim=-1, keepdim=True))
    fourmomenta = torch.cat([E, p3], dim=-1)

    # determine transformation into rest frame
    rest_trafo = restframe_transform(fourmomenta)
    fourmomenta_rest = torch.einsum("...ij,...j->...i", rest_trafo, fourmomenta)

    # check that the transformed fourmomenta are in the rest frame,
    # i.e. their spatial components vanish and the temporal component is the mass
    torch.testing.assert_close(
        fourmomenta_rest[..., 1:], torch.zeros_like(p3), **TOLERANCES
    )
    torch.testing.assert_close(
        fourmomenta_rest[..., 0] ** 2, lorentz_norm(fourmomenta), **TOLERANCES
    )
