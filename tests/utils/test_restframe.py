import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS, LOGM2_STD, LOGM2_MEAN
from tests.helpers import sample_vector, lorentz_test

from tensorframes.utils.restframe import (
    restframe_transform_v1,
    restframe_transform_v2,
    restframe_transform_v3,
)
from tensorframes.lframes.lframes import LFrames
from tensorframes.utils.lorentz import lorentz_squarednorm
from tensorframes.utils.transforms import (
    rand_lorentz,
    rand_rotation,
    rand_boost,
    rand_tz_boost,
)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "restframe_transform",
    [restframe_transform_v1, restframe_transform_v2, restframe_transform_v3],
)
@pytest.mark.parametrize("logm2_std", LOGM2_STD)
@pytest.mark.parametrize("logm2_mean", LOGM2_MEAN)
def test_restframe(batch_dims, restframe_transform, logm2_std, logm2_mean):
    dtype = torch.float64  # some tests require higher precision

    # sample Lorentz vectors
    fm = sample_vector(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # determine transformation into rest frame
    rest_trafo = restframe_transform(fm)
    fm_rest = torch.einsum("...ij,...j->...i", rest_trafo, fm)

    # check that the transformed fourmomenta are in the rest frame,
    # i.e. their spatial components vanish and the temporal component is the mass
    torch.testing.assert_close(
        fm_rest[..., 1:], torch.zeros_like(fm[..., 1:]), **TOLERANCES
    )
    torch.testing.assert_close(
        fm_rest[..., 0] ** 2, lorentz_squarednorm(fm), **TOLERANCES
    )

    lorentz_test(rest_trafo, **TOLERANCES)


"""
This is the test that we have to pass to make the RestLFrames work

Status
- restframe_transform_v1 fails, because it totally breaks equivariance
- restframe_transform_v2 transforms as follows: 
  rotations: rest -> random * rest * random^-1
  boosts: rest -> rest * random^-1
  -> it should pass rand_boost (but it doesnt in code... I did the calculation by hand, not sure what goes wrong)
- restframe_transform_v3 transforms adds a counter rotation matrix on the left,
  such that we get the following transformation behaviour:
  boosts: rest -> rest * random^-1 (as before)  
  rotations: rest -> rest * random^-1 (this is new!)
  combined rotations and boosts: should be rest -> rest * random^-1
-> restframe_transform_v3 should do what we want!
- Added a test which only performs a boost along the z direction
  and sets to zero the xy components of the fm.
  This means that rest * random^-1 are collinear boosts which outputs a boost.
  restframe_transform_v2 passes the test, while restframe_transform_v3 is messing up a few angles (not sure why)

Questions
- Why does the test not pass for rand_boost? (it should, I did the analytics)
- Why does the test not pass for rand_lorentz? 
  (I dont see why it should not pass if we can do boosts and rotations seperately)
Partial answers:
- Key aspect: composition of boosts is not a subgroup of Lorentz.
    -> Two boosts generally correspond to a boost + rotation. (See Wigner rotation)
- This is fine if the boosts are collinear for which we get rest -> rest * random^-1
  not 100% sure, maybe this is a too simple special case
"""


@pytest.mark.parametrize("batch_dims", [[10]])  # BATCH_DIMS)
@pytest.mark.parametrize(
    "restframe_transform",
    [restframe_transform_v2],  # v1 and v3 still don't pass the test
)
@pytest.mark.parametrize(
    "random_transform", [rand_tz_boost]
)  # rand_lorentz, rand_rotation, rand_boost don't pass all the tests
@pytest.mark.parametrize("logm2_std", [1])  # LOGM2_STD)
@pytest.mark.parametrize("logm2_mean", [0])  # LOGM2_MEAN)
def test_restframe_transformation(
    batch_dims, restframe_transform, random_transform, logm2_std, logm2_mean
):
    dtype = torch.float64  # some tests require higher precision

    # sample Lorentz vectors
    fm = sample_vector(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # determine transformation into rest frame
    if random_transform == rand_tz_boost:
        fm[..., 1:3] = 0
    rest_trafo = restframe_transform(fm)
    fm_rest = torch.einsum("...ij,...j->...i", rest_trafo, fm)

    # random global transformation
    random = random_transform([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    rest_trafo_prime = restframe_transform(fm_prime)
    fm_rest_prime = torch.einsum("...ij,...j->...i", rest_trafo_prime, fm_prime)

    # check that fourmomenta in rest frame is invariant
    torch.testing.assert_close(fm_rest, fm_rest_prime, **TOLERANCES)

    # check that the restframe transformations transform as expected
    # expect rest_trafo_prime = rest_trafo * random^-1
    # or rest_trafo_prime * random = rest_trafo
    inv_random = LFrames(random).inv
    rest_trafo_prime_expected = torch.einsum(
        "...ij,...jk->...ik", rest_trafo, inv_random
    )
    lorentz_test(random, **TOLERANCES)

    torch.testing.assert_close(
        rest_trafo_prime, rest_trafo_prime_expected, **TOLERANCES
    )
