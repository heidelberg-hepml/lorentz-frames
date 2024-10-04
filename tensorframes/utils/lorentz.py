import torch

from typing import Optional


def ltensor(a: torch.Tensor, b: torch.Tensor, dim: int = 0):
    index = [slice(None)] * b.dim()

    index[dim] = 0

    b_copy = b.detach().clone()
    b_copy[index] *= -1

    return torch.tensordot(a, b_copy, [[dim], [dim]])


def leinsum(a: torch.Tensor, b: torch.Tensor, dim: int = 0, einstr="i,i"):
    index = [slice(None)] * b.dim()

    index[dim] = 0

    b_copy = b.detach().clone()
    b_copy[index] *= -1
    return torch.einsum(einstr, a, b_copy)


def lnorm(a, dim: int = 0, einstr="i,i"):
    return torch.sqrt(torch.abs(leinsum(a, a, dim=dim, einstr=einstr)))

    return torch.tensordot(a, b_copy, [[dim], [dim]])
