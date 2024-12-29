import torch

EPS = 1e-10
CUTOFF = 10


def unpack_last(x):
    # unpack along the last dimension
    n = len(x.shape)
    return torch.permute(x, (n - 1, *list(range(n - 1))))


def stable_arctanh(x, eps=EPS):
    # implementation of arctanh that avoids log(0) issues
    return 0.5 * (torch.log((1 + x).clamp(min=eps)) - torch.log((1 - x).clamp(min=eps)))


def EPPP_to_PtPhiEtaM2(fourmomenta, sqrt_mass=False):
    E, px, py, pz = unpack_last(fourmomenta)
    pt = torch.sqrt(px**2 + py**2)
    phi = torch.arctan2(py, px)
    p_abs = torch.sqrt(pz**2 + pt**2)
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
