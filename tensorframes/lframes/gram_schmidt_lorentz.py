import warnings
from typing import Union

import torch
from torch import Tensor

# from tensorframes.utils.lorentz import lnorm, ldo
from typing import Optional

from tensorframes.utils.lorentz import lnorm, leinsum


def orthogonalize(vectors, newVector, eps):
    subtract = 0

    for i in range(vectors.shape[1]):
        subtract += (
            leinsum(newVector, vectors[:, i], 1, "ik,ik->i").unsqueeze(-1)
            * vectors[:, i]
            * leinsum(vectors[:, i], vectors[:, i], 1, "ij,ij->i").unsqueeze(
                -1
            )  # this is a bit odd
        )

    newVector -= subtract
    newNorm = lnorm(newVector, 1, "ij,ij->i")

    newVector[newNorm < eps] = float("nan")
    newVector[newNorm > eps] /= newNorm[newNorm > eps].unsqueeze(-1)

    return newVector


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

    assert vectors.shape[1:] == (4, 4)
    assert normalized == True

    lenghts = lnorm(vectors, dim=2, einstr="ijk,ijk->ij")
    vectors[lenghts < eps] = torch.rand((lenghts < eps).sum(), 4)
    lenghts = lnorm(vectors, dim=2, einstr="ijk,ijk->ij")

    vectors[:, 0] /= lenghts[:, 0].unsqueeze(-1)

    i = 1
    while True:
        newVector = orthogonalize(vectors[:, :i], vectors[:, i], eps)
        nanmask = torch.isnan(newVector)[:, 0]
        if torch.sum(nanmask) != 0:
            vectors[nanmask, i] = torch.rand(torch.sum(nanmask), 4)
        else:
            i += 1

        if i >= 4:
            break

    return vectors
