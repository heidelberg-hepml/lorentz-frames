"""Equivariance (output-invariance) tests for the graph-transformer tagging models.

Uses ``tests/helpers/equivariance.py`` to transform the input four-momenta by
random group elements and assert the classification score is unchanged. Three
properties are checked:

1. **Azimuthal (xy-rotation) invariance** for *every* graph-transformer hybrid.
   A rotation about the beam is the residual symmetry that survives the beam/time
   spurions, so it must hold for all of them -- the LLoCa-canonicalized backbones
   (ParticleNet, plain) just as much as the internally-equivariant ones (CGENN,
   LorentzNet). We build the kNN graph with the Minkowski metric here so the graph
   does not inherit the ``deltaR`` branch cut (phi from ``atan2`` jumps at +-pi).
   A static kNN graph is still only *approximately* invariant -- float32 re-ranks
   near-tied neighbours, so an edge can flip and shift the score by ~1e-3 -- which
   is inherent to every kNN GNN (ParticleNet included), so the tolerance is loose.

2. **Full SO(3) / Lorentz invariance** of the internally-equivariant hybrids
   (CGENN, LorentzNet) once their symmetry-breaking input spurions and the
   (non-invariant) global tagging features are switched off. Here we make the
   graph *fully connected* (``k=null`` / a large ``knn_k``) to remove the kNN
   discontinuity, and run in float64 (``use_float64=true``) so the geometric
   products keep their precision under boosts -- the backbone is then equivariant
   to ~1e-7, which is the real claim this test pins down.

3. **LLoCa frame invariance** of the *non-equivariant* backbones (ParticleNet,
   plain) under a learned Lorentz frame (``learnedso13``). This is LLoCa's central
   claim for them: the learned local frame canonicalizes every particle, so the
   kNN graph *and* all features are built from frame-invariant quantities and the
   score is invariant under the full Lorentz group -- to ~1e-7, even with the
   default kNN (no neighbour flips, because the graph is built on local momenta).

These run on the small ``config_quick`` models so they stay fast on CPU.
"""

import hydra
import pytest

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment
from tests.helpers.equivariance import check_tagging_invariance


def _build(overrides):
    experiments.logger.LOGGER.disabled = True
    with hydra.initialize(config_path="../../config_quick", version_base=None):
        cfg = hydra.compose(
            config_name="toptagging",
            overrides=[
                "save=false",
                "training.batchsize=8",
                "data.dataset=mini",
                *overrides,
            ],
        )
    exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    return exp


# ---------------------------------------------------------------------------
# 1. azimuthal invariance: the residual symmetry of every tagging setup
# ---------------------------------------------------------------------------
GRAPH_TRANS_MODELS = [
    "tag_CGENNLGATrGraphTrans",
    "tag_LorentzNetLGATrSlimGraphTrans",
    "tag_ParticleNetParTGraphTrans",
    "tag_PlainGraphTrans",
    "tag_PlainGraphGPS",
    "tag_ParticleNetParTGraphGPS",
    "tag_CGENNLGATrGraphGPS",
    "tag_LorentzNetLGATrSlimGraphGPS",
]


@pytest.mark.parametrize("model", GRAPH_TRANS_MODELS)
def test_xyrotation_invariance(model):
    # Minkowski kNN -> no deltaR branch cut (see module docstring). The residual
    # ~1e-3 comes only from static-kNN neighbour flips, hence the 1e-2 tolerance.
    exp = _build([f"model={model}", "model.net.knn_metric=minkowski"])
    data = next(iter(exp.train_loader))
    max_dev = check_tagging_invariance(
        exp, data, transform="xyrotation", num_checks=5, rtol=1e-2, atol=1e-2
    )
    assert max_dev < 2e-2, f"{model}: not xy-rotation invariant (max dev {max_dev:.2e})"


# ---------------------------------------------------------------------------
# 2. full-group invariance of the internally-equivariant hybrids
# ---------------------------------------------------------------------------
# With their input spurions and the (non-invariant) global tagging features
# removed, and a fully connected graph (no kNN discontinuity), CGENN/LorentzNet
# are equivariant under the whole group -> an invariant score.
FULL_GROUP_MODELS = [
    (
        "tag_CGENNLGATrGraphGPS",
        [
            "model.net.beam_spurion=null",
            "model.net.add_time_spurion=false",
            "model.net.k=null",  # fully connected: no kNN neighbour-flip discontinuity
        ],
    ),
    (
        "tag_LorentzNetLGATrSlimGraphGPS",
        [
            "model.net.use_time_spurion=false",
            "model.net.use_beam_spurion=false",
            "model.net.knn_k=9999",  # >= P-1 -> fully connected
        ],
    ),
    (
        "tag_CGENNLGATrGraphTrans",
        [
            "model.net.beam_spurion=null",
            "model.net.add_time_spurion=false",
            "model.net.k=null",  # fully connected: no kNN neighbour-flip discontinuity
        ],
    ),
    (
        "tag_LorentzNetLGATrSlimGraphTrans",
        [
            "model.net.use_time_spurion=false",
            "model.net.use_beam_spurion=false",
            "model.net.knn_k=9999",  # >= P-1 -> fully connected
        ],
    ),
]


@pytest.mark.parametrize("model,full_group_off", FULL_GROUP_MODELS)
@pytest.mark.parametrize("transform", ["rotation", "lorentz"])
def test_full_group_invariance(model, full_group_off, transform):
    exp = _build(
        [
            f"model={model}",
            "use_float64=true",  # geometric products lose float32 precision under boosts
            "data.tagging_features=null",
            "data.beam_reference=null",
            "data.add_time_reference=false",
            *full_group_off,
        ]
    )
    data = next(iter(exp.train_loader))
    max_dev = check_tagging_invariance(
        exp, data, transform=transform, num_checks=5, rtol=1e-4, atol=1e-4
    )
    assert max_dev < 1e-3, f"{model}: not {transform} invariant (max dev {max_dev:.2e})"


# ---------------------------------------------------------------------------
# 3. LLoCa frame invariance of the non-equivariant backbones
# ---------------------------------------------------------------------------
# A learned local frame -- LLoCa's default polar decomposition (learnedpd) or the
# orthonormal SO(1,3) tetrad (learnedso13) -- canonicalizes every particle, so the kNN
# graph and all features are frame-invariant and the (non-equivariant) backbone becomes
# Lorentz invariant. Both frame families are exercised because they feed jet_frames
# different ortho_kwargs (learnedpd: eps_reg -> the 3d name; learnedso13: eps_reg_coplanar).
# Beam/time references and the global tagging features are switched off so nothing else
# breaks the symmetry; learned frames run in float64.
#
# The ParticleNet-ParT hybrids now do genuine LLoCa tensorial message-passing (neighbours
# transported in the EdgeConv, q/k/v transported in the attention), so this pins down the
# transport, not just input canonicalization -- exact to ~1e-6. ParticleNetParTGraphGPS
# rebuilds a *dynamic* kNN graph every layer (DGCNN-style), so under a boost near-tied
# neighbours re-rank and the graph jumps (the ~1e-3 kNN floor of test 1, amplified by the
# tensorial transport). We test its transport fully connected to isolate it from that
# discontinuity, which test 1 covers separately.
CANONICALIZED_MODELS = [
    ("tag_ParticleNetParTGraphTrans", []),
    ("tag_PlainGraphTrans", []),
    ("tag_PlainGraphGPS", []),
    ("tag_ParticleNetParTGraphGPS", ["model.net.knn_k=9999"]),  # dynamic kNN -> fully connected
]


# Per-frame float64 tolerance: (rtol, atol, bound). learnedso13 builds the frame by direct
# 4d orthonormalization and transports near-exactly (~1e-6). learnedpd's polar decomposition
# divides by energy in the rest-frame boost, so boosts amplify rounding and the per-edge
# transport accumulates a higher floor (worse with more edges; ~5e-3 fully connected). Both
# are exact in real arithmetic; the looser learnedpd bound still catches any true break
# (those are O(0.1-1), far above the floor).
FRAME_TOL = {
    "learnedso13": (1e-4, 1e-4, 1e-3),
    "learnedpd": (1e-2, 1e-2, 2e-2),
}


@pytest.mark.parametrize("model,extra", CANONICALIZED_MODELS)
@pytest.mark.parametrize("framesnet", ["learnedpd", "learnedso13"])
@pytest.mark.parametrize("transform", ["rotation", "lorentz"])
def test_lloca_frame_invariance(model, transform, extra, framesnet):
    exp = _build(
        [
            f"model={model}",
            f"model/framesnet={framesnet}",
            "use_float64=true",
            "data.tagging_features=null",
            "data.beam_reference=null",
            "data.add_time_reference=false",
            *extra,
        ]
    )
    data = next(iter(exp.train_loader))
    rtol, atol, bound = FRAME_TOL[framesnet]
    max_dev = check_tagging_invariance(
        exp, data, transform=transform, num_checks=5, rtol=rtol, atol=atol
    )
    assert max_dev < bound, (
        f"{model}/{framesnet}: LLoCa frame not {transform} invariant (max dev {max_dev:.2e})"
    )
