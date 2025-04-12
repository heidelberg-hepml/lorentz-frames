import os
from numpy import load
import torch

from experiments.amplitudes.constants import get_mass

from tensorframes.utils.lorentz import lorentz_eye
from tensorframes.utils.transforms import (
    rand_rotation_uniform,
    rand_xyrotation,
)
from tensorframes.utils.restframe import restframe_boost


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


def load_file(
    data_path,
    cfg_data,
    dataset,
    amp_mean=None,
    amp_std=None,
    mom_std=None,
    dtype=torch.float32,
    generator=None,
):
    assert os.path.exists(data_path)
    data_raw = load(data_path)
    data_raw = torch.tensor(data_raw, dtype=dtype)

    if cfg_data.subsample is not None:
        assert cfg_data.subsample <= data_raw.shape[0]
        data_raw = data_raw[: cfg_data.subsample]

    momentum = data_raw[:, :-1]
    momentum = momentum.reshape(momentum.shape[0], momentum.shape[1] // 4, 4)
    amplitude = data_raw[:, [-1]]

    # mass regulator
    if cfg_data.mass_reg is not None:
        mass = get_mass(dataset, cfg_data.mass_reg)
        mass = torch.tensor(mass, dtype=dtype).unsqueeze(0)
        momentum[..., 0] = torch.sqrt((momentum[..., 1:] ** 2).sum(dim=-1) + mass**2)

    # prepare momenta
    if cfg_data.prepare == "centerofmass":
        # rotation in z-direction to go to center-of-mass frame
        lab_momentum = momentum[..., :2, :].sum(dim=-2)
        trafo = restframe_boost(-lab_momentum)
    elif cfg_data.prepare == "lorentz":
        # add random rotation to existing z-boost -> general Lorentz trafo
        trafo = rand_rotation_uniform(
            momentum.shape[:-2], generator=generator, dtype=dtype
        )
    elif cfg_data.prepare == "ztransform":
        # add random xyrotation to existing z-boost -> general ztransform
        trafo = rand_xyrotation(momentum.shape[:-2], generator=generator, dtype=dtype)
    elif cfg_data.prepare == "identity":
        trafo = lorentz_eye(momentum.shape[:-2], device=momentum.device, dtype=dtype)
    else:
        raise ValueError(f"cfg.data.prepare={cfg_data.prepare} not implemented")
    momentum = torch.einsum("...ij,...kj->...ki", trafo, momentum)

    if mom_std is None:
        mom_std = momentum.std()
    momentum /= mom_std

    amplitude, amp_mean, amp_std = preprocess_amplitude(
        amplitude, std=amp_std, mean=amp_mean
    )
    return amplitude, momentum, amp_mean, amp_std, mom_std
