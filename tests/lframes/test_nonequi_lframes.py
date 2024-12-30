import torch
import pytest
from tests.constants import TOLERANCES

from tensorframes.reps import TensorReps
from tensorframes.lframes.nonequi_lframes import (
    IdentityLFrames,
    RandomLFrames,
    RandomPhiLFrames,
)


@pytest.mark.parametrize(
    "LFramesPredictor", [IdentityLFrames, RandomLFrames, RandomPhiLFrames]
)
@pytest.mark.parametrize("shape", [[1000]])
def test_constructor(LFramesPredictor, shape):
    fourmomenta = torch.randn(*shape, 4)

    # predict local frames
    predictor = LFramesPredictor()
    lframes = predictor(fourmomenta)

    # transform into local frames
    reps = TensorReps("1x1n")
    trafo = TensorReps(reps).get_transform_class()
    fourmomenta_local = trafo(fourmomenta, lframes)

    if LFramesPredictor == IdentityLFrames:
        # fourmomenta should not change
        torch.testing.assert_close(fourmomenta_local, fourmomenta, **TOLERANCES)
    elif LFramesPredictor == RandomPhiLFrames:
        # energy and pz should not change
        torch.testing.assert_close(
            fourmomenta_local[..., [0, 3]], fourmomenta[..., [0, 3]], **TOLERANCES
        )
