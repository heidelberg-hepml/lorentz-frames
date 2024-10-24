import warnings
from typing import Union

import torch
from torch import Tensor

from typing import Optional

# this could probably be moved to the /utils path eventually, together with a method for the norm
def leinsum(einstr: str, a: torch.Tensor, b: torch.Tensor, dim: int = -1):
    index = [slice(None)] * b.dim()

    index[dim] = 0

    b_copy = b.detach().clone()
    b_copy[index] *= -1
    return torch.einsum(einstr, a, b_copy)


def gram_schmidt_lorentz(
    vectors,
    eps: float = 1e-6,
    normalized: bool = True,
    exceptional_choice: str = "random",
) -> Tensor:
    """Applies the Gram-Schmidt process to a set of input vectors to orthogonalize them.

    Args:
        vectors (Tensor): The input vectors. shape (N, 4, 4) (batch, vectors, dims)
        eps (float, optional): A small value used for numerical stability. Defaults to 1e-6.
        normalized (bool, optional): Whether to normalize the output vectors. Defaults to True.
        exceptional_choice (str, optional): The method to handle exceptional cases where the input vectors have zero length.
            Can be either "random" to use a random vector instead, or "zero" to set the vectors to zero.
            Defaults to "random".

    Returns:
        Tensor: A tensor containing the orthogonalized vectors the tensor has shape (N, 4, 4).

    Raises:
        ValueError: If the exceptional_choice parameter is not recognized.
        AssertionError: If z_axis has zero length.
    """
    vectors = vectors.clone()
    assert vectors.shape[1:] == (4, 4)
    assert normalized == True

    errorCounter = 0
    for index in range(vectors.shape[1]):
        error = True
        while error == True:
            sign = (
                leinsum("nmd,nmd->nm", vectors[:, :index], vectors[:, :index], dim=-1)
                .sign()
                .unsqueeze(-1)
            )
            weights = leinsum(
                "npf,nf->np", sign * vectors[:, :index], vectors[:, index], dim=-1
            )
            normBefore = (
                leinsum("nd,nd->n", vectors[:, index], vectors[:, index], dim=-1)
                .abs()
                .sqrt()
            )
            vectors[:, index] -= torch.sum(
                torch.einsum("nm,nmj->nmj", weights, vectors[:, :index]), axis=1
            )
            norm = (
                leinsum("nd,nd->n", vectors[:, index], vectors[:, index], dim=-1)
                .abs()
                .sqrt()
            )
            zeroNorm = norm < 2.0e-1 * normBefore
            if zeroNorm.sum().item() != 0:  # linearly alligned elements / zero norm
                vectors[zeroNorm, index] = torch.rand(vectors[zeroNorm, index].shape)
                errorCounter += 1
            else:
                vectors[:, index] /= norm.unsqueeze(1)
                error = False
    return vectors
