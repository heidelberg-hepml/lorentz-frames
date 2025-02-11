import torch
import pytest
from tests.constants import MILD_TOLERANCES, BATCH_DIMS, LOGM2_MEAN_STD
from tests.helpers import sample_particle

from tensorframes.utils.hep import EPPP_to_PtPhiEtaM2, PtPhiEtaM2_to_EPPP


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_invertibility(batch_dims, logm2_std, logm2_mean):
    dtype = torch.float32

    EPPP = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    PtPhiEtaM2 = EPPP_to_PtPhiEtaM2(EPPP)
    EPPP_reconstructed = PtPhiEtaM2_to_EPPP(PtPhiEtaM2)

    torch.testing.assert_close(EPPP, EPPP_reconstructed, **MILD_TOLERANCES)
