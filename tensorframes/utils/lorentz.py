import torch


def lorentz_inner(v1, v2):
    """Lorentz inner product, i.e v1*g*v2"""
    prod = v1 * v2
    prod *= torch.tensor([1, -1, -1, -1], device=v1.device, dtype=v1.dtype)
    return prod.sum(dim=-1)


def lorentz_norm(v):
    """Lorentz norm, i.e. v*g*v"""
    return lorentz_inner(v, v)


def leinsum(einstr: str, a: torch.Tensor, b: torch.Tensor, dim: int = -1):
    """torch.einsum, but uses the minkovski metric (1, -1, -1, -1)
    e.g.
    a = torch.tensor([1,2,1,2])
    b = torch.tensor([[2,2,2,2]])
    result = leinsum(einstr="d,bd->b", a, b, dim=-1)

    will calculate the following:
        result = torch.einsum(einstr="d,bd->b", a, torch.tensor([[2,-2,-2,-2]]))

    Args:
        einstr (str): string for einstein notations
        a, b (tensors): tensors to operate on
        dim (int): dimention in which the first element should have opposite sign

    Returns:
        einsum of the tensors
    """
    index = [slice(None)] * b.dim()

    index[dim] = slice(1, None)

    b_copy = b.detach().clone()
    b_copy[index] *= -1
    return torch.einsum(einstr, a, b_copy)
