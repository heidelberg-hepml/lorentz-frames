import torch

from .orthogonalize_o3 import orthogonalize_o3
from .lorentz import lorentz_squarednorm


def restframe_boost(fourmomenta):
    """
    Lorentz transformation representing a boost into the rest frame.
    This transformation does not have the lframes transformation properties,
    because there is no rotation aspect - it is simply a boost.

    Args:
        fourmomenta: torch.tensor of shape (*dims, 4)

    Returns:
        trafo: torch.tensor of shape (*dims, 4, 4)
    """
    assert (
        lorentz_squarednorm(fourmomenta) > 0
    ).all(), "Trying to boost spacelike vectors into their restframe (not possible). Consider changing the nonlinearity in equivectors."

    beta = fourmomenta[..., 1:] / fourmomenta[..., [0]].clamp(min=1e-10)
    beta2 = (beta**2).sum(dim=-1, keepdim=True)
    gamma = 1 / (1 - beta2).clamp(min=1e-10).sqrt()

    # prepare entries of the trafo
    boost = -gamma * beta
    eye = torch.eye(3, device=fourmomenta.device, dtype=fourmomenta.dtype)
    eye = eye.view(*(1,) * len(fourmomenta.shape[:-1]), 3, 3).repeat(
        *fourmomenta.shape[:-1], 1, 1
    )
    rot = eye + (
        (gamma[..., None] - 1)
        * beta[..., None]
        * beta[..., None, :]
        / beta2[..., None].clamp(min=1e-10)
    )

    # put trafo together
    trafo = torch.empty(
        *fourmomenta.shape[:-1],
        4,
        4,
        device=fourmomenta.device,
        dtype=fourmomenta.dtype
    )
    trafo[..., 0, 0] = gamma[..., 0]
    trafo[..., 1:, 1:] = rot
    trafo[..., 0, 1:] = boost
    trafo[..., 1:, 0] = boost
    assert torch.isfinite(trafo).all()
    return trafo


def restframe_equivariant(
    fourmomenta, references, use_float64=True, return_reg=False, **kwargs
):
    """
    Lorentz transformation representing a boost into the rest frame and a
    properly constructed rotation that fixes the little group degree of freedom
    The resulting transformation has the lframes transformation behaviour

    Args:
        fourmomenta: torch.tensor of shape (*dims, 4)
            Four-momentum that defines the rest frames
        references: List with two torch.tensor of shape (*dims, 4)
            Two reference four-momenta to construct the rotation
        use_float64: bool
            Use float64 for calculations?
        return_reg: bool
            Return dict with regularization information?
        **kwargs: Additional arguments for orthogonalize_o3

    Returns:
        trafo: torch.tensor of shape (*dims, 4, 4)
    """
    assert len(references) == 2
    assert all(r.shape == fourmomenta.shape for r in references)

    if use_float64:
        original_dtype = fourmomenta.dtype
        fourmomenta = fourmomenta.to(torch.float64)
        references = [r.to(torch.float64) for r in references]

    # construct rest frame transformation
    boost = restframe_boost(fourmomenta)

    # references go into rest frame
    ref_rest = [torch.einsum("...ij,...j->...i", boost, v) for v in references]

    # construct rotation
    ref3_rest = [r[..., 1:] for r in ref_rest]
    out = orthogonalize_o3(ref3_rest, return_reg=return_reg, **kwargs)
    if return_reg:
        orthogonal_vec3, reg_collinear = out
    else:
        orthogonal_vec3 = out
    rotation = torch.zeros_like(boost)
    rotation[..., 0, 0] = 1
    rotation[..., 1:, 1:] = torch.stack(orthogonal_vec3, dim=-2)

    # combine rotation and boost
    trafo = torch.einsum("...ij,...jk->...ik", rotation, boost)
    if use_float64:
        trafo = trafo.to(original_dtype)
    assert torch.isfinite(trafo).all()
    return (trafo, reg_collinear) if return_reg else trafo
