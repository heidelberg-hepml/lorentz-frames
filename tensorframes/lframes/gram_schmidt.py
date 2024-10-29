import warnings
from typing import Union

import torch


def leinsum(einstr: str, a: torch.Tensor, b: torch.Tensor, dim: int = -1):
    """torch.einsum, but inverts the first element in the dimention dim

    Args:
        einstr (str): string for einstein notations
        a, b (tensors): tensors to operate on
        dim (int): dimention in which the first element should have opposite sign

    Returns:
        einsum of the tensors
    """
    index = [slice(None)] * b.dim()

    index[dim] = 0

    b_copy = b.detach().clone()
    b_copy[index] *= -1
    return torch.einsum(einstr, a, b_copy)


def gram_schmidt(
    vectors,
    eps: float = 2.0e-1,
    normalized: bool = True,
    exceptional_choice: str = "random",
) -> torch.Tensor:
    """Applies the Gram-Schmidt process to a set of input vectors to orthogonalize them.

    Args:
        vectors (Tensor): The input vectors. shape (4, 4, N) or (3, 4, N) (vectors, dims, size)
        eps (float, optional): A small value used for numerical stability. Defaults to 2.0e-1.
        normalized (bool, optional): Whether to normalize the output vectors. Defaults to True.
        exceptional_choice (str, optional): The method to handle exceptional cases where the input vectors have zero length.
            Can be either "random" to use a random vector instead, or "zero" to set the vectors to zero.
            Defaults to "random".

    Returns:
        Tensor: A tensor containing the orthogonalized vectors the tensor has shape (4, 4, N).
    """

    assert normalized == True
    assert (
        exceptional_choice == "random" or exceptional_choice == "zero"
    ), "Exception Choice needs to be 'zero' or 'random'"
    device = vectors.device

    assert vectors.shape[:-1] == (4, 4) or vectors.shape[:-1] == (
        3,
        4,
    ), f"The shape of the given vectors ({vectors.shape}) needs to be in the format (vectors, dim, size), where vectors can be 4 or 3, dim has to be 4 and size is arbitrary"
    for index in range(vectors.shape[0]):
        error = True
        while error == True:
            previous = vectors[:index].clone()
            sign = (
                leinsum("vds,vds->vs", previous, previous, dim=-2).sign().unsqueeze(-2)
            )
            weights = leinsum(
                "vds,ds->vs", sign * previous, vectors[index].clone(), dim=-2
            )
            normBefore = (
                leinsum("ds,ds->s", vectors[index], vectors[index], dim=-2).abs().sqrt()
            )
            vectors[index] -= torch.sum(
                torch.einsum("vs,vds->vds", weights, previous), axis=0
            )
            norm = (
                leinsum("ds,ds->s", vectors[index], vectors[index], dim=-2).abs().sqrt()
            )
            zeroNorm = norm < eps * normBefore
            if zeroNorm.sum().item() != 0:  # linearly alligned elements / zero norm
                if exceptional_choice == "random":
                    vectors[index, :, zeroNorm] = torch.rand(
                        vectors[index, :, zeroNorm].shape, device=device
                    )
                elif exceptional_choice == "zero":
                    vectors[index, :, zeroNorm] = torch.zeros(
                        vectors[index, :, zeroNorm].shape, device=device
                    )
            else:
                vectors[index] /= norm.unsqueeze(-2)
                error = False
    if vectors.shape[:-1] == (3, 4):
        vectors = torch.cat(
            (
                vectors,
                torch.zeros(1, vectors.shape[1], vectors.shape[-1], device=device),
            ),
            dim=0,
        )
        error = True
        while error:
            x, y, z, _ = torch.clamp(vectors.clone(), 1e-6)
            alpha1 = y[1] / x[1]
            alpha2 = z[1] / x[1]
            beta = (z[2] - alpha2 * x[2]) / (y[2] - alpha1 * x[2])

            vectors[-1, 3] = (
                (z[0] - alpha2 * x[0]) - beta * (y[0] - alpha1 * x[0])
            ) / ((z[3] - alpha2 * x[3]) - beta * (y[3] - alpha1 * x[3]))
            vectors[-1, 2] = (
                (y[0] - alpha1 * x[0]) - (y[3] - alpha1 * x[3]) * vectors[-1, 3].clone()
            ) / (y[2] - alpha1 * x[2])
            vectors[-1, 1] = (
                x[0] - x[2] * vectors[-1, 2].clone() - x[3] * vectors[-1, 3].clone()
            ) / x[1]
            vectors[-1, 0] = torch.tensor(1).repeat(vectors.shape[-1])
            norm = leinsum("ds,ds->s", vectors[-1], vectors[-1], dim=-2).abs().sqrt()
            vectors[-1] /= norm.unsqueeze(-2)

    return vectors.permute(2, 0, 1)
