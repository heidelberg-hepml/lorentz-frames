import torch
import pytest
from tests.constants import TOLERANCES, LOGM2_MEAN, LOGM2_STD
from tests.helpers import sample_vector

from tensorframes.reps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.lframes.nonequi_lframes import (
    IdentityLFrames,
    RandomLFrames,
    RandomPhiLFrames,
)


@pytest.mark.parametrize(
    "LFramesPredictor", [IdentityLFrames, RandomLFrames, RandomPhiLFrames]
)
@pytest.mark.parametrize("batch_dims", [[1000]])
@pytest.mark.parametrize("logm2_std", LOGM2_STD)
@pytest.mark.parametrize("logm2_mean", LOGM2_MEAN)
def test_vectors(LFramesPredictor, batch_dims, logm2_mean, logm2_std):
    dtype = torch.float32

    fm = sample_vector(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # predict local frames
    predictor = LFramesPredictor()
    lframes = predictor(fm)

    # transform into local frames
    reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(reps))
    fm_local = trafo(fm, lframes)

    if LFramesPredictor == IdentityLFrames:
        # fourmomenta should not change
        torch.testing.assert_close(fm_local, fm, **TOLERANCES)
    elif LFramesPredictor == RandomPhiLFrames:
        # energy and pz should not change
        torch.testing.assert_close(fm_local[..., [0, 3]], fm[..., [0, 3]], **TOLERANCES)
