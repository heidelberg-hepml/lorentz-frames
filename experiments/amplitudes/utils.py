import numpy as np

from tensorframes.utils.lorentz import lorentz_squarednorm
from experiments.amplitudes.constants import IN_PARTICLES


def preprocess_momentum(momentum, mean=None, std=None):
    # use common mean() and std() for all components in E, px, py, pz
    # otherwise std_E=std_pz=10*std_px=10*std_py
    # and get large values from data augmentations because e.g. pz->px
    if mean is None or std is None:
        mean = momentum.mean()
        std = momentum.std().clamp(min=1e-2)

    momentum_prepd = (momentum - mean) / std
    return momentum_prepd, mean, std


def preprocess_amplitude(amplitude, std=None, mean=None):
    log_amplitude = np.log(amplitude)
    if std is None or mean is None:
        mean = log_amplitude.mean()
        std = log_amplitude.std()
    prepd_amplitude = (log_amplitude - mean) / std
    return prepd_amplitude, mean, std


def undo_preprocess_amplitude(prepd_amplitude, mean, std):
    assert mean is not None and std is not None
    log_amplitude = prepd_amplitude * std + mean
    amplitude = np.exp(log_amplitude.clamp(min=-10, max=10))
    return amplitude
