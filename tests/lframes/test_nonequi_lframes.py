import torch
import pytest
from tests.constants import TOLERANCES, LOGM2_MEAN_STD
from tests.helpers import sample_particle

from tensorframes.reps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.lframes.nonequi_lframes import (
    IdentityLFrames,
    RandomLFrames,
)


@pytest.mark.parametrize(
    "LFramesPredictor,transform_type",
    [
        (IdentityLFrames, None),
        (RandomLFrames, "lorentz"),
        (RandomLFrames, "rotation"),
        (RandomLFrames, "boost"),
        (RandomLFrames, "xyrotation"),
    ],
)
@pytest.mark.parametrize("batch_dims", [[1000]])
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_vectors(LFramesPredictor, transform_type, batch_dims, logm2_mean, logm2_std):
    dtype = torch.float32

    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # predict local frames
    predictor = (
        LFramesPredictor(transform_type=transform_type)
        if LFramesPredictor == RandomLFrames
        else LFramesPredictor()
    )
    lframes = predictor(fm)

    # transform into local frames
    reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(reps))
    fm_local = trafo(fm, lframes)

    if LFramesPredictor == IdentityLFrames:
        # fourmomenta should not change
        torch.testing.assert_close(fm_local, fm, **TOLERANCES)
    elif type == "rotation":
        # energy and pz should not change
        torch.testing.assert_close(fm_local[..., [0]], fm[..., [0]], **TOLERANCES)
    elif type == "xyrotation":
        # energy and pz should not change
        torch.testing.assert_close(fm_local[..., [0, 3]], fm[..., [0, 3]], **TOLERANCES)
