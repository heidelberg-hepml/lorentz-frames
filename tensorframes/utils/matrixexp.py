import torch
from torch.linalg import matrix_exp

from tensorframes.utils.lorentz import lorentz_metric


def build_generator(v1, v2):
    # build antisymmetric generator
    generator = v1[..., None] * v2[..., None, :] - v1[..., None, :] * v2[..., None]

    # turn generator into transformation matrix
    metric = lorentz_metric(v1.shape[:-1], dtype=v1.dtype, device=v1.device)
    generator = generator @ metric
    return generator


def matrix_exponential(v1, v2):
    generator = build_generator(v1, v2)

    # carefully evaluate matrix exponential
    # caution: large generator matrices can cause infs
    assert (
        generator.max() < 1e2
    ), f"large generator matrices can cause infs in matrix_exp: generator.max()={generator.max()}."
    f"consider dividing v1, v2 by a sufficiently large number to avoid this."
    trafo = matrix_exp(generator)
    assert torch.isfinite(trafo).all()
    return trafo
