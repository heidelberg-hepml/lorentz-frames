import pytest
import hydra
from torch.utils.flop_counter import FlopCounterMode

import experiments.logger
from experiments.amplitudes.experiment import AmplitudeExperiment
from tests_exp.utils import fix_seeds


@pytest.mark.parametrize("lframesnet", ["identity", "polardec"])
@pytest.mark.parametrize(
    "model_list",
    [
        # ["model=amp_mlp"],
        ["model=amp_transformer"],
        ["model=amp_graphnet"],
        # ["model=amp_graphnet", "model.include_edges=false"],
        # ["model=amp_graphnet", "model.include_nodes=false"],
        ["model=amp_gatr"],
        ["model=amp_dsi"],
    ],
)
def test_amplitudes(lframesnet, model_list):
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
    try:
        exp.init_model()
    except Exception as e:
        return
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()

    data = next(iter(exp.train_loader))
    mom = data[1].to(device=exp.device, dtype=exp.dtype)

    with FlopCounterMode(display=False) as flop_counter:
        exp.model(mom)
    flops = flop_counter.get_total_flops()

    print(
        f"flops (batchsize=1): {flops:.2e}",
        model_list,
        lframesnet,
    )
    num_parameters = sum(p.numel() for p in exp.model.parameters())
    print(
        f"parameters: {num_parameters}",
    )
    print(flop_counter.get_table(depth=2))
