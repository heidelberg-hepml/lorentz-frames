import warnings
from typing import Union

import torch


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


def gram_schmidt(
    vectors,
    eps: float = 2.0e-1,
    normalized_last: bool = True,
    exceptional_choice: str = "random",
) -> torch.Tensor:
    """Applies the Gram-Schmidt process to a set of input vectors to orthogonalize them.

    Args:
        vectors (Tensor): The input vectors. shape (N, 4, 4) or (N, 3, 4) (size, vectors, dims)
        eps (float, optional): A small value used for numerical stability. Defaults to 2.0e-1.
        normalized_last (bool, optional): Whether to normalize the last vector when using the cross product to get it.
        exceptional_choice (str, optional): The method to handle exceptional cases where the input vectors have zero length.
            Can be either "random" to use a random vector instead, or "zero" to set the vectors to zero.
            Defaults to "random".

    Returns:
        Tensor: A tensor containing the orthogonalized vectors the tensor has shape (N, 4, 4).
    """

    assert (
        exceptional_choice == "random" or exceptional_choice == "zero"
    ), "Exception Choice needs to be 'zero' or 'random'"
    device = vectors.device

    assert vectors.shape[1:] == (4, 4) or vectors.shape[1:] == (
        3,
        4,
    ), f"The shape of the given vectors ({vectors.shape}) needs to be in the format (size, vector, dim), where vectors can be 4 or 3, dim has to be 4 and size is arbitrary"
    orthogonalized_vectors = vectors.clone()
    for index in range(vectors.shape[-2]):
        error = True
        while error == True:
            previous = orthogonalized_vectors[:, :index].clone()
            sign = (
                leinsum("svd,svd->sv", previous, previous, dim=-1).sign().unsqueeze(-1)
            )
            weights = leinsum(
                "svd,sd->sv",
                sign * previous,
                orthogonalized_vectors[:, index].clone(),
                dim=-1,
            )
            orthogonalized_vectors[:, index] -= torch.sum(
                torch.einsum("sv,svd->svd", weights, previous), axis=-2
            )
            norm = (
                leinsum(
                    "sd,sd->s",
                    orthogonalized_vectors[:, index],
                    orthogonalized_vectors[:, index],
                    dim=-1,
                )
                .abs()
                .sqrt()
            )
            zeroNorm = norm < eps
            if zeroNorm.sum().item() != 0:  # linearly alligned elements / zero norm
                if exceptional_choice == "random":
                    orthogonalized_vectors[zeroNorm, index, :] += (
                        torch.rand(
                            orthogonalized_vectors[zeroNorm, index, :].shape,
                            device=device,
                        )
                        - 0.5
                    )
                elif exceptional_choice == "zero":
                    orthogonalized_vectors[zeroNorm, index, :] = torch.zeros(
                        orthogonalized_vectors[zeroNorm, index, :].shape, device=device
                    )
            else:
                orthogonalized_vectors[:, index] /= norm.unsqueeze(-1) + 1e-6
                error = False
    if vectors.shape[1:] == (3, 4):
        orthogonalized_vectors = torch.cat(
            (
                orthogonalized_vectors,
                torch.zeros(
                    orthogonalized_vectors.shape[0],
                    1,
                    orthogonalized_vectors.shape[-1],
                    device=device,
                ),
            ),
            dim=1,
        )
        x, y, z, _ = orthogonalized_vectors.clone().unbind(dim=1)
        orthogonalized_vectors[:, -1] = torch.stack(
            [
                -x[:, 1] * y[:, 2] * z[:, 3]
                + x[:, 1] * y[:, 3] * z[:, 2]
                - x[:, 2] * y[:, 3] * z[:, 1]
                + x[:, 2] * y[:, 1] * z[:, 3]
                - x[:, 3] * y[:, 1] * z[:, 2]
                + x[:, 3] * y[:, 2] * z[:, 1],
                -x[:, 0] * y[:, 2] * z[:, 3]
                + x[:, 0] * y[:, 3] * z[:, 2]
                - x[:, 2] * y[:, 3] * z[:, 0]
                + x[:, 2] * y[:, 0] * z[:, 3]
                - x[:, 3] * y[:, 0] * z[:, 2]
                + x[:, 3] * y[:, 2] * z[:, 0],
                +x[:, 0] * y[:, 1] * z[:, 3]
                - x[:, 0] * y[:, 3] * z[:, 1]
                + x[:, 1] * y[:, 3] * z[:, 0]
                - x[:, 1] * y[:, 0] * z[:, 3]
                + x[:, 3] * y[:, 0] * z[:, 1]
                - x[:, 3] * y[:, 1] * z[:, 0],
                -x[:, 0] * y[:, 1] * z[:, 2]
                + x[:, 0] * y[:, 2] * z[:, 1]
                - x[:, 1] * y[:, 2] * z[:, 0]
                + x[:, 1] * y[:, 0] * z[:, 2]
                - x[:, 2] * y[:, 0] * z[:, 1]
                + x[:, 2] * y[:, 1] * z[:, 0],
            ],
            dim=-1,
        )
        norm = (
            leinsum(
                "sd,sd->s",
                orthogonalized_vectors[:, -1],
                orthogonalized_vectors[:, -1],
                dim=-1,
            )
            .abs()
            .sqrt()
        )
        if normalized_last:
            orthogonalized_vectors[:, -1] /= norm.unsqueeze(-1) + 1e-6

    return orthogonalized_vectors


def is_orthogonal(vec):
    """checks if vectors are orthogonal

    args:
        vec (torch.Tensor): vectors in shape (N, 4, 4)
    returns:
        is_orthogonal (torch.Tensor): bool list of orthogonal tests
        prod (torch.Tensor): tensor of product between vectors
    """
    vec2 = vec.clone()
    vec2[:, :, 0] *= -1

    prod = torch.round(torch.einsum("ijk,imk->ijm", vec, vec2).abs().sqrt(), decimals=2)
    return (
        (prod - torch.eye(4).unsqueeze(0).repeat(prod.shape[0], 1, 1)).abs() < eps
    ).all(dim=(1, 2))
