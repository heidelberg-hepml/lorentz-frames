import pytest
import hydra
from torch.utils.flop_counter import FlopCounterMode

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment


@pytest.mark.parametrize(
    "lframesnet",
    [
        "identity",
        "randomrotation",
        "randomlorentz",
        "orthogonal",
        "polardec",
    ],
)
@pytest.mark.parametrize(
    "model_list",
    [
        ["model=tag_ParT"],
        ["model=tag_particlenet-lite"],
        ["model=tag_transformer"],
        ["model=tag_graphnet"],
        ["model=tag_graphnet", "model.include_edges=true"],
    ],
)
@pytest.mark.parametrize("iterations", [1])
def test_tagging(lframesnet, model_list, iterations):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    # note: use the full models
    with hydra.initialize(config_path="../config", version_base=None):
        overrides = [
            *model_list,
            f"model/lframesnet={lframesnet}",
            "save=false",
            "training.batchsize=1",
        ]
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
        exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()

    for i, data in enumerate(exp.train_loader):
        if i == iterations:
            break

        with FlopCounterMode(display=False) as flop_counter:
            exp._get_ypred_and_label(data)
            flops = flop_counter.get_total_flops()

    print(
        f"flops (batchsize=1): {flops:.2e}",
        model_list,
        lframesnet,
    )
