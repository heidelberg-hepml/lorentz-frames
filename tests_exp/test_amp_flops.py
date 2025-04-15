import pytest
import hydra
from torch.utils.flop_counter import FlopCounterMode

import experiments.logger
from experiments.amplitudes.experiment import AmplitudeExperiment
from tests_exp.utils import fix_seeds


@pytest.mark.parametrize("lframesnet", ["identity"])
@pytest.mark.parametrize(
    "model_list",
    [
        ["model=amp_mlp"],
        ["model=amp_transformer"],
        ["model=amp_graphnet"],
        ["model=amp_graphnet", "model.include_edges=false"],
        ["model=amp_graphnet", "model.include_nodes=false"],
        ["model=amp_gatr"],
        ["model=amp_dsi"],
    ],
)
@pytest.mark.parametrize("iterations", [1])
def test_amplitudes(lframesnet, model_list, iterations):
    experiments.logger.LOGGER.disabled = True  # turn off logging
    fix_seeds(0)
    # create experiment environment
    with hydra.initialize(config_path="../config", version_base=None):
        overrides = [
            *model_list,
            f"model/lframesnet={lframesnet}",
            "save=false",
            "training.batchsize=1",
        ]
        cfg = hydra.compose(config_name="amplitudes", overrides=overrides)
        exp = AmplitudeExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()

    for i, data in enumerate(exp.train_loader):
        mom = data[1].to(device=exp.device, dtype=exp.dtype)
        if i == iterations:
            break
        with FlopCounterMode(display=False) as flop_counter:
            exp.model(mom)
            flops = flop_counter.get_total_flops()

    print(
        f"flops (batchsize=1): {flops:.2e}",
        model_list,
        lframesnet,
    )
