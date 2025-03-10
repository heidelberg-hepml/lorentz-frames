import os
from numpy import load
import torch

from experiments.amplitudes.constants import get_mass

from tensorframes.utils.transforms import (
    rand_lorentz,
    rand_rotation,
    rand_xyrotation,
    rand_boost,
)


def standardize_momentum(momentum, mean=None, std=None):
    # use common mean() and std() for all components in E, px, py, pz
    # note: empirically this step is not super important; rescaling by momentum.std() does the job
    if mean is None or std is None:
        mean = momentum.mean()
        std = momentum.std().clamp(min=1e-2)

    momentum_prepd = (momentum - mean) / std
    return momentum_prepd, mean, std


def preprocess_amplitude(amplitude, std=None, mean=None):
    log_amplitude = amplitude.log()
    if std is None or mean is None:
        mean = log_amplitude.mean()
        std = log_amplitude.std()
    prepd_amplitude = (log_amplitude - mean) / std
    return prepd_amplitude, mean, std


def undo_preprocess_amplitude(prepd_amplitude, mean, std):
    assert mean is not None and std is not None
    log_amplitude = prepd_amplitude * std + mean
    amplitude = log_amplitude.clamp(max=10).exp()
    return amplitude


def load_file(data_path, cfg_data, dataset, momentum_std=None, dtype=torch.float32):
    assert os.path.exists(data_path)
    data_raw = load(data_path)
    data_raw = torch.tensor(data_raw, dtype=dtype)

    momentum = data_raw[:, :-1]
    momentum = momentum.reshape(momentum.shape[0], momentum.shape[1] // 4, 4)
    amplitude = data_raw[:, [-1]]

    # mass regulator
    if cfg_data.mass_reg is not None:
        mass = get_mass(dataset, cfg_data.mass_reg)
        mass = torch.tensor(mass, dtype=dtype).unsqueeze(0)
        momentum[..., 0] = (momentum[..., 1:] ** 2).sum(dim=-1) + cfg_data.mass_reg**2

    # prepare momenta
    if cfg_data.prepare == "align":
        # momenta are already aligned with the beam
        pass
    else:
        if cfg_data.prepare == "xyrotation":
            trafo = rand_xyrotation(momentum.shape[:-2])
        elif cfg_data.prepare == "rotation":
            trafo = rand_rotation(momentum.shape[:-2])
        elif cfg_data.prepare == "boost":
            trafo = rand_boost(momentum.shape[:-2])
        elif cfg_data.prepare == "lorentz":
            trafo = rand_lorentz(momentum.shape[:-2])
        else:
            raise ValueError(f"cfg.data.prepare={cfg_data.prepare} not implemented")
        momentum = torch.einsum("...ij,...kj->...ki", trafo, momentum)

    if momentum_std is None:
        momentum_std = momentum.std()
    momentum /= momentum_std
    return amplitude, momentum, momentum_std
