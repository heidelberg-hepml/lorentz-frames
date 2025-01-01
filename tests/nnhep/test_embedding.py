import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS, LOGM2_MEAN, LOGM2_STD
from tests.helpers import sample_vector

from tensorframes.nnhep.embedding import EPPP_to_PtPhiEtaM2, PtPhiEtaM2_to_EPPP


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("logm2_std", LOGM2_STD)
@pytest.mark.parametrize("logm2_mean", LOGM2_MEAN)
def test_invertibility(batch_dims, logm2_std, logm2_mean):
    dtype = torch.float32

    EPPP = sample_vector(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    PtPhiEtaM2 = EPPP_to_PtPhiEtaM2(EPPP)
    EPPP_reconstructed = PtPhiEtaM2_to_EPPP(PtPhiEtaM2)

    torch.testing.assert_close(EPPP, EPPP_reconstructed, **TOLERANCES)
