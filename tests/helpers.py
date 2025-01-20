import torch
import os
import numpy as np
from functools import lru_cache

from tensorframes.utils.lorentz import lorentz_metric


def sample_vector(
    shape, logm2_std, logm2_mean, device=torch.device("cpu"), dtype=torch.float32
):
    assert logm2_std > 0
    logm2 = torch.randn(*shape, 1, device=device, dtype=dtype) * logm2_std + logm2_mean
    p3 = torch.randn(*shape, 3, device=device, dtype=dtype)
    E = torch.sqrt(logm2.exp() + (p3**2).sum(dim=-1, keepdim=True))
    return torch.cat([E, p3], dim=-1)


def lorentz_test(trafo, **kwargs):
    """
    Test that the transformation matrix T is orthogonal

    Condition: T^T * g * T = g
    with the Lorentz metric g = diag(1, -1, -1, -1)
    """
    metric = lorentz_metric(trafo.shape[:-2], trafo.device, trafo.dtype)
    test = torch.einsum(
        "...ij,...jk,...kl->...il", trafo, metric, trafo.transpose(-1, -2)
    )
    torch.testing.assert_close(test, metric, **kwargs)


@lru_cache()
def load_data():
    # load particles from the toptagging_mini.npz dataset
    # this function is cached to avoid loading the data multiple times
    filename = os.path.join("data", "toptagging_mini.npz")
    assert os.path.exists(filename), f"File not found: {filename}"
    data = np.load(filename)
    fourmomenta = data["kinematics_train"] / 20  # rescaled fourmomenta
    fourmomenta = torch.tensor(fourmomenta)
    mask = (fourmomenta.abs() > 1e-5).all(dim=-1)
    fourmomenta = fourmomenta[mask]
    return fourmomenta


def sample_vector_realistic(shape, device=torch.device("cpu"), dtype=torch.float32):
    # sample from the cached toptagging_mini.npz dataset
    full = load_data()
    idx = torch.randint(0, full.shape[0], shape)
    return full[idx].to(dtype=dtype, device=device)
