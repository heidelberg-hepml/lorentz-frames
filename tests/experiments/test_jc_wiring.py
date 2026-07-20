"""JetClass-path wiring test: compose -cn jctagging for every GT hybrid and assert
init_physics wires the extra-scalars channels. Composition-level only (no data, no
forward): this is the layer where this repo has shipped broken twice (tag_lorentznet's
in_s_channels/n_scalar key, jc_lgatr's tag_gatr base) because nothing exercised the
jctagging config path."""
import hydra, pytest, logging
import experiments.logger; experiments.logger.LOGGER.disabled = True; logging.disable(logging.CRITICAL)
from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment

HYBRIDS_NONEQ = ["tag_PlainGraphTrans", "tag_PlainGraphGPS",
                 "tag_ParticleNetParTGraphTrans", "tag_ParticleNetParTGraphGPS"]
HYBRIDS_EQ = ["tag_CGENNLGATrGraphTrans", "tag_CGENNLGATrGraphGPS",
              "tag_LorentzNetLGATrSlimGraphTrans", "tag_LorentzNetLGATrSlimGraphGPS"]

@pytest.mark.parametrize("model", HYBRIDS_NONEQ + HYBRIDS_EQ)
def test_jc_wiring(model):
    with hydra.initialize(config_path="../../config", version_base=None):
        cfg = hydra.compose(config_name="jctagging", overrides=[f"model={model}", "save=false"])
    exp = JetClassTaggingExperiment(cfg)   # features=default -> extra_scalars=10
    exp.init_physics()                     # THE wiring under test (twice-bitten layer)
    if model in HYBRIDS_NONEQ:
        assert cfg.model.in_channels == 7 + 10, cfg.model.in_channels
    else:
        assert cfg.model.net.in_s_channels == 10 + 7, cfg.model.net.in_s_channels
    assert cfg.model.out_channels == 10
