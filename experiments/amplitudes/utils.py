import numpy as np

from tensorframes.utils.lorentz import lorentz_squarednorm


def preprocess_momentum(momentum, mean=None, std=None):
    if mean is None or std is None:
        mean = momentum.mean(dim=[0, 1])
        std = momentum.std(dim=[0, 1]).clamp(min=1e-2)

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


def encode_event(fourmomenta, n_in=2):
    fourmomenta_in = fourmomenta[..., :n_in, :]
    fourmomenta_out = fourmomenta[..., n_in:, :]

    initial_state = fourmomenta_in.sum(dim=-2)
    in_invariant = lorentz_squarednorm(initial_state)
    in_invariant = in_invariant.clamp(min=1e-10).log()
    return in_invariant, fourmomenta_out
