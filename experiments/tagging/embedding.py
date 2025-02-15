import torch
import math
from torch_geometric.utils import scatter, dense_to_sparse

from tensorframes.utils.hep import get_eta, get_phi, get_pt
from experiments.tagging.dataset import EPS

UNITS = 20  # We use units of 20 GeV for all tagging experiments

# Preprocessing (mean, std) for scalar features
# these are the defaults used in weaver
SCALAR_FEATURES_PREPROCESSING = [
    [1.7 - math.log(UNITS), 0.7],  # log_pt
    [2.0 - math.log(UNITS), 0.7],  # log_energy
    [-4.7 - math.log(UNITS), 0.7],  # log_pt_rel
    [-4.7 - math.log(UNITS), 0.7],  # log_energy_rel
    [0, 1],  # dphi
    [0, 1],  # deta
    [0.2, 4],  # dr
]


def get_batch_from_ptr(ptr):
    return torch.arange(len(ptr) - 1, device=ptr.device).repeat_interleave(
        ptr[1:] - ptr[:-1],
    )


def embed_tagging_data(fourmomenta, scalars, ptr, cfg_data):
    """
    Embed tagging data
    We use torch_geometric sparse representations to be more memory efficient
    Note that we do not embed the label, because it is handled elsewhere

    Parameters
    ----------
    fourmomenta: torch.tensor of shape (n_particles, 4)
        Fourmomenta in the format (E, px, py, pz)
    scalars: torch.tensor of shape (n_particles, n_features)
        Optional scalar features, n_features=0 is possible
    ptr: torch.tensor of shape (batchsize+1)
        Indices of the first particle for each jet
        Also includes the first index after the batch ends
    cfg_data: settings for embedding

    Returns
    -------
    embedding: dict
        Embedded data
        Includes keys for fourmomenta (n_particle (+n_spurion if beam_token), n_vectors, 4), scalars and ptr
    """
    batchsize = len(ptr) - 1
    arange = torch.arange(batchsize, device=fourmomenta.device)

    # add mass regulator
    if cfg_data.mass_reg is not None:
        fourmomenta[..., 0] = (
            (fourmomenta[..., 1:] ** 2).sum(dim=-1) + cfg_data.mass_reg**2
        ).sqrt()

    # add extra scalar channels
    if cfg_data.add_scalar_features:
        features = get_tagging_features(fourmomenta, ptr, cfg_data)
        scalars = torch.cat(
            (scalars, features),
            dim=-1,
        )

    if cfg_data.rescale_data:
        fourmomenta /= UNITS

    # beam reference
    spurions = get_spurion(
        cfg_data.beam_reference,
        cfg_data.add_time_reference,
        cfg_data.two_beams,
        fourmomenta.device,
        fourmomenta.dtype,
    )

    fourmomenta = fourmomenta.unsqueeze(1)
    n_spurions = spurions.shape[0]
    assert cfg_data.beam_token, f"spurions as channels not consistently supported yet"
    if cfg_data.beam_token and n_spurions > 0:
        # prepend spurions to the token list (within each block)
        spurion_idxs = torch.stack(
            [ptr[:-1] + i for i in range(n_spurions)], dim=0
        ) + n_spurions * torch.arange(batchsize, device=ptr.device)
        spurion_idxs = spurion_idxs.permute(1, 0).flatten()
        insert_spurion = torch.zeros(
            fourmomenta.shape[0] + n_spurions * batchsize,
            dtype=torch.bool,
            device=fourmomenta.device,
        )
        insert_spurion[spurion_idxs] = True
        fourmomenta_buffer = fourmomenta.clone()
        fourmomenta = torch.empty(
            insert_spurion.shape[0],
            *fourmomenta.shape[1:],
            dtype=fourmomenta.dtype,
            device=fourmomenta.device,
        )
        fourmomenta[~insert_spurion] = fourmomenta_buffer
        fourmomenta[insert_spurion] = spurions.repeat(batchsize, 1).unsqueeze(1)
        scalars_buffer = scalars.clone()
        scalars = torch.zeros(
            fourmomenta.shape[0],
            scalars.shape[1],
            dtype=scalars.dtype,
            device=scalars.device,
        )
        scalars[~insert_spurion] = scalars_buffer
        ptr[1:] = ptr[1:] + (arange + 1) * n_spurions
    else:
        # append spurion to fourmomenta channels
        spurions = spurions.unsqueeze(0).repeat(fourmomenta.shape[0], 1, 1)
        fourmomenta = torch.cat((fourmomenta, spurions), dim=-2)

    # construct edge_index (dark art)
    diffs = torch.diff(ptr)
    edge_index = torch.cat(
        [
            dense_to_sparse(torch.ones(d, d, device=diffs.device))[0] + diffs[:i].sum()
            for i, d in enumerate(diffs)
        ],
        dim=-1,
    )

    # return dict
    batch = get_batch_from_ptr(ptr)
    embedding = {
        "fourmomenta": fourmomenta,
        "scalars": scalars,
        "edge_index": edge_index,
        "batch": batch,
    }
    return embedding


def dense_to_sparse_jet(fourmomenta_dense, scalars_dense):
    """
    Transform dense jet into sparse jet

    Parameters
    ----------
    fourmomenta_dense: torch.tensor of shape (batchsize, 4, num_particles_max)
    scalars_dense: torch.tensor of shape (batchsize, num_features, num_particles_max)

    Returns
    -------
    fourmomenta_sparse: torch.tensor of shape (num_particles, 4)
        Fourmomenta for concatenated list of particles of all jets
    scalars_sparse: torch.tensor of shape (num_particles, num_features)
        Scalar features for concatenated list of particles of all jets
    ptr: torch.tensor of shape (batchsize+1)
        Start indices of each jet, this way we don't lose information when concatenating everything
        Starts with 0 and ends with the first non-accessible index (=total number of particles)
    """
    fourmomenta_dense = torch.transpose(
        fourmomenta_dense, 1, 2
    )  # (batchsize, num_particles, 4)
    scalars_dense = torch.transpose(
        scalars_dense, 1, 2
    )  # (batchsize, num_particles, num_features)

    mask = (fourmomenta_dense.abs() > EPS).any(dim=-1)
    num_particles = mask.sum(dim=-1)
    fourmomenta_sparse = fourmomenta_dense[mask]
    scalars_sparse = scalars_dense[mask]

    ptr = torch.zeros(
        len(num_particles) + 1, device=fourmomenta_dense.device, dtype=torch.long
    )
    ptr[1:] = torch.cumsum(num_particles, dim=0)
    return fourmomenta_sparse, scalars_sparse, ptr


def get_spurion(
    beam_reference,
    add_time_reference,
    two_beams,
    device,
    dtype,
):
    """
    Construct spurion

    Parameters
    ----------
    beam_reference: str
        Different options for adding a beam_reference
    add_time_reference: bool
        Whether to add the time direction as a reference to the network
    two_beams: bool
        Whether we only want (x, 0, 0, 1) or both (x, 0, 0, +/- 1) for the beam
    device
    dtype

    Returns
    -------
    spurion: torch.tensor with shape (n_spurions, 4)
        spurion embedded as fourmomenta object
    """

    if beam_reference in ["lightlike", "spacelike", "timelike"]:
        # add another 4-momentum
        if beam_reference == "lightlike":
            beam = [1, 0, 0, 1]
        elif beam_reference == "timelike":
            beam = [2**0.5, 0, 0, 1]
        elif beam_reference == "spacelike":
            beam = [0, 0, 0, 1]
        beam = torch.tensor(beam, device=device, dtype=dtype).reshape(1, 4)
        if two_beams:
            beam2 = beam.clone()
            beam2[..., 3] = -1  # flip pz
            beam = torch.cat((beam, beam2), dim=0)

    elif beam_reference is None:
        beam = torch.empty(0, 4, device=device, dtype=dtype)

    else:
        raise ValueError(f"beam_reference {beam_reference} not implemented")

    if add_time_reference:
        time = [1, 0, 0, 0]
        time = torch.tensor(time, device=device, dtype=dtype).reshape(1, 4)
    else:
        time = torch.empty(0, 4, device=device, dtype=dtype)

    spurion = torch.cat((beam, time), dim=-2)
    return spurion


def get_tagging_features(fourmomenta, ptr, cfg_data):
    """
    Compute features typically used in jet tagging

    Parameters
    ----------
    fourmomenta: torch.tensor of shape (n_particles, 4)
        Fourmomenta in the format (E, px, py, pz)
    ptr: torch.tensor of shape (batchsize+1)
        Indices of the first particle for each jet
        Also includes the first index after the batch ends
    cfg_data: settings for embedding

    Returns
    -------
    features: torch.tensor of shape (n_particles, n_features)
        Features: log_pt, log_energy, log_pt_rel, log_energy_rel, dphi, deta, dr
    """
    log_pt = get_pt(fourmomenta).unsqueeze(-1).log()
    log_energy = fourmomenta[..., 0].unsqueeze(-1).log()

    batch = get_batch_from_ptr(ptr)
    jet = scatter(fourmomenta, index=batch, dim=0, reduce="sum").index_select(0, batch)
    log_pt_rel = (get_pt(fourmomenta).log() - get_pt(jet).log()).unsqueeze(-1)
    log_energy_rel = (fourmomenta[..., 0].log() - jet[..., 0].log()).unsqueeze(-1)
    phi_4, phi_jet = get_phi(fourmomenta), get_phi(jet)
    dphi = ((phi_4 - phi_jet + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)
    eta_4, eta_jet = get_eta(fourmomenta), get_eta(jet)
    deta = -(eta_4 - eta_jet).unsqueeze(-1)
    dr = torch.sqrt(dphi**2 + deta**2)
    features = [
        log_pt,
        log_energy,
        log_pt_rel,
        log_energy_rel,
        dphi,
        deta,
        dr,
    ]
    for i, feature in enumerate(features):
        mean, factor = SCALAR_FEATURES_PREPROCESSING[i]
        features[i] = (feature - mean) * factor
    return torch.cat(features, dim=-1)
