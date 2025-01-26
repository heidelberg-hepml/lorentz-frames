from itertools import pairwise
import torch

from tensorframes.utils.lorentz import (
    lorentz_inner,
    lorentz_squarednorm,
    leinsum,
    lorentz_metric,
)

from tensorframes.utils.hep import get_deltaR


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
    eps: float = 1.0e-6,
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


def regularize_lightlike(
    vecs: torch.tensor,
    exception_eps: float = 1e-8,
    sample_eps: float = 1.0e-7,
    rejection_regularize=False,
):
    """
    Regularize the inputs to avoid lightlike vectors
    Args:
        vecs (Tensor): torch tensor of shape (3, N, 4) with N the batch dimension
        exception_eps (float): threshold applied to the criterion
        sample_eps (float): rescaling applied to the sampled four vectors
    Returns:
        tensor: regularized four vectors
    """
    if rejection_regularize:
        inners = torch.stack([lorentz_inner(v, v) for v in vecs])
        mask = (inners.abs() < exception_eps).to(vecs.device)[:, 0]
        return vecs[torch.argsort(mask)]

    assert vecs.shape[0] == 3

    inners = torch.stack([lorentz_inner(v, v) for v in vecs])
    sample = sample_eps * torch.randn(vecs.shape, dtype=vecs.dtype, device=vecs.device)
    mask = (inners.abs() < exception_eps)[..., None].expand_as(sample)
    vecs = vecs + sample * mask

    return vecs


def regularize_collinear(
    vecs: torch.tensor,
    exception_eps: float = 1e-6,
    sample_eps: float = 1.0e-5,
    rejection_regularize=False,
):
    """
    Regularize the inputs to avoid collinear vectors
    Args:
        vecs (Tensor): torch tensor of shape (3, N, 4) with N the batch dimension
        exception_eps (float): threshold applied to the criterion
        sample_eps (float): rescaling applied to the sampled four vectors
    Returns:
        tensor: regularized four vectors
    """
    if rejection_regularize:
        error = True
        safety = 10
        while error and (safety := safety - 1) > 0:
            error = False
            v_pairs = torch.cat((vecs[:3], vecs[0][None, ...]))
            deltaRs = torch.stack([get_deltaR(v, vp) for v, vp in pairwise(v_pairs)])
            mask = deltaRs < exception_eps
            if mask.sum() != 0:
                error = True
            mask = torch.cat(
                (
                    mask,
                    torch.full(
                        (vecs.shape[0] - 3, *mask.shape[1:]), True, device=vecs.device
                    ),
                ),
                dim=0,
            )

            indices = torch.argmax(mask.to(int), dim=0).tolist()
            temp = vecs[indices, torch.arange(vecs.shape[-2])]
            vecs[indices, torch.arange(vecs.shape[-2])] = vecs[3].clone()
            vecs[3:] = torch.cat((vecs[4:], temp.unsqueeze(0)))
        return vecs

    assert vecs.shape[0] == 3

    v_pairs = torch.cat((vecs, vecs[0][None, ...]))
    deltaRs = torch.stack([get_deltaR(v, vp) for v, vp in pairwise(v_pairs)])
    sample = sample_eps * torch.randn(vecs.shape, dtype=vecs.dtype, device=vecs.device)
    mask = (deltaRs < exception_eps)[..., None].expand_as(sample)
    vecs = vecs + sample * mask

    return vecs


def regularize_coplanar(
    vecs: torch.tensor,
    exception_eps: float = 1e-7,
    sample_eps: float = 1.0e-6,
    rejection_regularize=False,
):
    """
    Regularize the inputs to avoid collinear vectors
    Args:
        vecs (Tensor): torch tensor of shape (3, N, 4) with N the batch dimension
        exception_eps (float): threshold applied to the criterion
        sample_eps (float): rescaling applied to the sampled four vectors
    Returns:
        tensor: regularized four vectors
    """
    if rejection_regularize:
        error = True
        safety = 10
        while error and (safety := safety - 1) > 0:
            error = False
            cross_norm = lorentz_squarednorm(lorentz_cross(vecs[0], vecs[1], vecs[2]))
            mask = cross_norm.abs() < exception_eps
            vecs[2:, mask] = torch.cat(
                (vecs[3:, mask], vecs[2, mask].unsqueeze(0)), dim=0
            )

            if mask.sum() != 0:
                error = True
        return vecs

    assert vecs.shape[0] == 3

    cross_norm = lorentz_squarednorm(lorentz_cross(vecs[0], vecs[1], vecs[2]))
    sample = sample_eps * torch.randn(vecs.shape, dtype=vecs.dtype, device=vecs.device)
    mask = (cross_norm.abs() < exception_eps)[None, :, None].expand_as(sample)
    vecs = vecs + sample * mask

    return vecs


def order_vectors(
    trafo,
):
    # sort vectors by norm -> first vector has >0 norm
    # this is necessary to get valid Lorentz transforms
    # see paper for why at most one vector can have >0 norm
    vecs = [trafo[..., i, :] for i in range(4)]
    norm = torch.stack([lorentz_squarednorm(v) for v in vecs], dim=-1)
    pos_norm = norm > 0
    num_pos_norm = pos_norm.sum(dim=-1)
    assert (
        num_pos_norm == 1
    ).all(), f"Warning: find different number of norm>0 vectors: {torch.unique(num_pos_norm)}"
    old_trafo = trafo.clone()
    trafo[..., 0, :] = old_trafo[pos_norm]
    trafo[..., 1:, :] = old_trafo[~pos_norm].view(-1, 3, 4)

    return trafo


def cross_trafo(vecs, regularize=True, rejection_regularize=False, eps=1e-10):
    if regularize:
        vecs = regularize_collinear(vecs, rejection_regularize=rejection_regularize)
        vecs = regularize_coplanar(vecs, rejection_regularize=rejection_regularize)
        vecs = regularize_lightlike(vecs, rejection_regularize=rejection_regularize)
    vecs = vecs[:3]

    orthogonal_vecs = orthogonalize_cross(vecs, eps)
    trafo = torch.stack(orthogonal_vecs, dim=-2)

    # turn into transformation matrix
    metric = lorentz_metric(trafo.shape[:-2], device=trafo.device, dtype=trafo.dtype)
    trafo = trafo @ metric
    trafo = order_vectors(trafo)

    return trafo
