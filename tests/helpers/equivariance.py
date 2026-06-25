"""Invariance / equivariance checks for the tagging models.

Inspired by the L-GATr equivariance helper
(https://github.com/heidelberg-hepml/lorentz-gatr/blob/main/tests/helpers/equivariance.py),
adapted to this repo: the tagging models output a classification score, which
must be **invariant** under the (residual) symmetry group. These helpers
transform the input four-momenta by random group elements and assert that the
score is unchanged within tolerance, over several random draws (``num_checks``).

The available transforms come from ``lloca.utils.rand_transforms``:
    xyrotation  -- azimuthal SO(2) about the beam (preserved by every tagging
                   setup, including those with beam/time spurions),
    rotation    -- SO(3),
    boost       -- a Lorentz boost,
    lorentz     -- the full SO(1,3).
"""

import torch
from lloca.utils.rand_transforms import (
    rand_boost,
    rand_lorentz,
    rand_rotation,
    rand_xyrotation,
)

TRANSFORMS = {
    "xyrotation": rand_xyrotation,
    "rotation": rand_rotation,
    "boost": rand_boost,
    "lorentz": rand_lorentz,
}

#maybe add z axis? 
def transform_momenta(data, trafo_fn, dtype):
    """Clone a tagging batch and apply a random ``trafo_fn`` to its four-momenta.

    One transform is drawn per jet and broadcast over its constituents, so the
    whole event is transformed coherently (matching tests/.../test_tag_invariance).
    """
    out = data.clone()
    mom = out.x
    trafo = trafo_fn(mom.shape[:-2] + (1,), dtype=dtype)
    out.x = torch.einsum("...ij,...j->...i", trafo, mom)
    return out


def check_tagging_invariance(
    experiment,
    data,
    transform="xyrotation",
    num_checks=5,
    rtol=1e-3,
    atol=1e-3,
    seed=42,
):
    """Assert the model's score is invariant under ``transform``.

    Runs the model on ``data`` and on ``num_checks`` random transforms of it,
    asserting the scores agree within ``(rtol, atol)`` via
    ``torch.testing.assert_close``. A fresh clone is used for every forward pass
    because ``_get_ypred_and_label`` mutates the batch (it shifts ``ptr`` when
    adding spurions). ``seed`` fixes the random group elements so the check is
    reproducible. Returns the largest absolute deviation observed.
    """
    trafo_fn = TRANSFORMS[transform] if isinstance(transform, str) else transform
    experiment.model.eval()
    torch.manual_seed(seed)
    max_dev = 0.0
    with torch.no_grad():
        y0 = experiment._get_ypred_and_label(data.clone())[0]
        for _ in range(num_checks):
            data_t = transform_momenta(data, trafo_fn, dtype=experiment.momentum_dtype)
            yt = experiment._get_ypred_and_label(data_t)[0]
            torch.testing.assert_close(yt, y0, rtol=rtol, atol=atol)
            max_dev = max(max_dev, (yt - y0).abs().max().item())
    return max_dev
