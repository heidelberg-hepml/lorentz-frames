import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS

from tensorframes.nnhep.embedding import EPPP_to_PtPhiEtaM2, PtPhiEtaM2_to_EPPP


@pytest.mark.parametrize("shape", BATCH_DIMS)
@pytest.mark.parametrize("logm2_std", [0.1, 1])
@pytest.mark.parametrize("logm2_mean", [-3, 0, 3])
def test_invertibility(shape, logm2_std, logm2_mean):
    p3 = torch.randn(shape + [3])
    logm2 = torch.randn(shape + [1]) * logm2_std + logm2_mean
    E = torch.sqrt(logm2.exp() + (p3**2).sum(dim=-1, keepdim=True))
    EPPP = torch.cat([E, p3], dim=-1)

    PtPhiEtaM2 = EPPP_to_PtPhiEtaM2(EPPP)
    EPPP_reconstructed = PtPhiEtaM2_to_EPPP(PtPhiEtaM2)

    torch.testing.assert_close(EPPP, EPPP_reconstructed, **TOLERANCES)
