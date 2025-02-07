import torch

from tensorframes.utils.utils import stable_arctanh, unpack_last

EPS = 1e-10
CUTOFF = 10


def EPPP_to_PtPhiEtaM2(fourmomenta, sqrt_mass=False):
    E, px, py, pz = unpack_last(fourmomenta)

    pt = torch.sqrt((px**2 + py**2).clamp(min=EPS))
    phi = torch.arctan2(avoid_zero(py), avoid_zero(px))
    p_abs = torch.sqrt((pz**2 + pt**2).clamp(min=EPS))
    eta = stable_arctanh(pz / p_abs).clamp(min=-CUTOFF, max=CUTOFF)
    m2 = E**2 - px**2 - py**2 - pz**2
    m2 = torch.sqrt(m2.clamp(min=EPS)) if sqrt_mass else m2
    return torch.stack((pt, phi, eta, m2), dim=-1)


def PtPhiEtaM2_to_EPPP(x):
    pt, phi, eta, m2 = unpack_last(x)
    eta = eta.clamp(min=-CUTOFF, max=CUTOFF)
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    E = torch.sqrt(m2 + pt**2 * torch.cosh(eta) ** 2)
    return torch.stack((E, px, py, pz), dim=-1)


def get_pt(p):
    # transverse momentum
    return torch.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)


def avoid_zero(x, eps=EPS):
    # set small-abs values to eps for numerical stability
    return torch.where(x.abs() < eps, eps, x)


def get_phi(p):
    # azimuthal angle
    return torch.arctan2(avoid_zero(p[..., 2]), avoid_zero(p[..., 1]))


def get_eta(p):
    # rapidity
    p_abs = torch.sqrt(torch.sum(p[..., 1:] ** 2, dim=-1))
    return stable_arctanh(p[..., 3] / p_abs)


def get_deltaR(v1, v2):
    # deltaR = sqrt((eta1-eta2)^2 + (phi1 - phi2)^2)
    eta1 = get_eta(v1)
    eta2 = get_eta(v2)

    phi1 = get_phi(v1)
    phi2 = get_phi(v2)

    delta_y = eta1 - eta2
    delta_phi = (phi1 - phi2 + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.sqrt(delta_y**2 + delta_phi**2)
