import torch

from tensorframes.utils.lorentz import lorentz_squarednorm, leinsum


def lorentz_cross(v1, v2, v3):
    """
    Compute the cross product in Minkowski space.

    Args:
        v1, v2, v3: Tensors of shape (*dims, 4)

    Returns:
        v4: Tensor of shape (*dims, )
    """
    assert v1.shape[-1] == 4
    assert v1.shape == v2.shape and v1.shape == v3.shape

    mat = torch.stack([v1, v2, v3], dim=-1)

    # euclidean fully antisymmetric product
    v4 = []
    for n in range(4):
        minor = torch.cat([mat[..., :n, :], mat[..., n + 1 :, :]], dim=-2)
        contribution = (-1) ** n * torch.det(minor)
        v4.append(contribution)
    v4 = torch.stack(v4, dim=-1)

    # raise indices with metric tensor
    v4 *= torch.tensor([1.0, -1.0, -1.0, -1.0], device=v1.device, dtype=v1.dtype)
    return v4


def orthogonalize_cross(vecs, eps=1e-10):
    n_vectors = len(vecs)
    assert n_vectors == 3

    def normalize(v):
        norm = lorentz_squarednorm(v).unsqueeze(-1)
        norm = norm.abs().sqrt()  # could also multiply by torch.sign(norm)
        return v / (norm + eps)

    vecs = [normalize(v) for v in vecs]

    # orthogonalize vectors with repeated cross products
    orthogonal_vecs = [vecs[0]]
    for i in range(1, n_vectors + 1):
        v_next = lorentz_cross(*orthogonal_vecs, *vecs[i:])
        assert torch.isfinite(v_next).all()
        orthogonal_vecs.append(normalize(v_next))

    return orthogonal_vecs


def orthogonalize_cross_o3(vecs, eps=1e-10):
    n_vectors = len(vecs)
    assert n_vectors == 2

    def normalize(v):
        norm = torch.linalg.norm(v, dim=-1, keepdim=True)
        return v / (norm + eps)

    vecs = [normalize(v) for v in vecs]

    orthogonal_vecs = [vecs[0]]
    for i in range(1, n_vectors + 1):
        v_next = torch.cross(*orthogonal_vecs, *vecs[i:], dim=-1)
        assert torch.isfinite(v_next).all()
        orthogonal_vecs.append(normalize(v_next))

    return orthogonal_vecs


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
