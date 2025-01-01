import torch
from math import factorial
from torch.linalg import matrix_exp

from tensorframes.utils.lorentz import lorentz_metric


def build_generator(v1, v2):
    # build antisymmetric generator
    generator = v1[..., None] * v2[..., None, :] - v1[..., None, :] * v2[..., None]

    # turn generator into transformation matrix
    metric = lorentz_metric(v1.shape[:-1], dtype=v1.dtype, device=v1.device)
    generator = generator @ metric
    return generator


def matrix_exponential(v1, v2, n_max=20):
    generator = build_generator(v1, v2)

    trafo = matrix_exp(generator)
    return trafo
