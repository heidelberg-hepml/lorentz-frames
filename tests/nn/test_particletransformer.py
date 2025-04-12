import torch
import pytest
from tests.constants import (
    TOLERANCES,
    MILD_TOLERANCES,
    LOGM2_MEAN_STD,
    REPS,
    LFRAMES_PREDICTOR,
)
from tests.helpers import sample_particle, equivectors_builder
from torch_geometric.utils import dense_to_sparse

from tensorframes.nn.particletransformer import ParticleTransformer
from tensorframes.reps.tensorreps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.utils.transforms import rand_lorentz
from tensorframes.lframes.equi_lframes import (
    LearnedOrthogonalLFrames,
    LearnedPolarDecompositionLFrames,
)

from experiments.tagging.embedding import get_tagging_features


@pytest.mark.parametrize(
    "LFramesPredictor", [LearnedOrthogonalLFrames, LearnedPolarDecompositionLFrames]
)  # RestLFrames gives nans sometimes
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("logm2_mean,logm2_std", [(0, 1)])  # LOGM2_MEAN_STD)
def test_particlenet_invariance(
    LFramesPredictor,
    batch_dims,
    logm2_std,
    logm2_mean,
):
    dtype = torch.float64
    batch = torch.zeros(batch_dims[0], dtype=torch.long)

    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = LFramesPredictor(equivectors=equivectors).to(dtype=dtype)
    call_predictor = lambda fm: predictor(fm)

    # define particlenet
    in_reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(in_reps))
    model = ParticleTransformer(
        input_dim=7,
        num_classes=1,
        attn_reps="8x0n+2x1n",
    ).to(dtype=dtype)
    model.eval()  # turn off dropout

    def ParT_wrapper(p_local, lframes):
        fts_local = get_tagging_features(p_local, batch)
        fts_local = fts_local.transpose(-1, -2).unsqueeze(0)
        p_local = p_local[..., [1, 2, 3, 0]]
        p_local = p_local.transpose(-1, -2).unsqueeze(0)
        mask = torch.ones_like(p_local[..., [0], :])
        lframes = lframes.reshape(1, *lframes.shape)
        x = model(x=fts_local, v=p_local, lframes=lframes, mask=mask)
        x = x.transpose(-1, -2).squeeze(0)
        return x

    # get global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    lframes = call_predictor(fm)
    fm_local = trafo(fm, lframes)

    # particlenet
    score_prime_local = ParT_wrapper(fm_local, lframes)

    # global - particlenet
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    lframes_transformed = call_predictor(fm_transformed)
    fm_tr_local = trafo(fm_transformed, lframes_transformed)
    score_tr_prime_local = ParT_wrapper(fm_tr_local, lframes_transformed)

    # test feature invariance before the operation
    torch.testing.assert_close(fm_local, fm_tr_local, **TOLERANCES)

    # test equivariance of scores
    torch.testing.assert_close(
        score_tr_prime_local, score_prime_local, **MILD_TOLERANCES
    )
