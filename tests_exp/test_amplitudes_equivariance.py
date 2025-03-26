import torch
import pytest
import hydra

import experiments.logger
from experiments.amplitudes.experiment import AmplitudeExperiment
from tensorframes.utils.transforms import rand_rotation, rand_lorentz


@pytest.mark.parametrize(
    "model_list",
    [
        ["model=amp_mlp"],
        ["model=amp_transformer"],
        ["model=amp_graphnet"],
        ["model=amp_graphnet", "model.include_edges=false"],
        ["model=amp_graphnet", "model.include_nodes=false"],
    ],
)
@pytest.mark.parametrize("lframesnet", ["orthogonal", "polardec"])
@pytest.mark.parametrize("rand_trafo", [rand_rotation, rand_lorentz])
@pytest.mark.parametrize("iterations", [1000])
def test_amplitudes(model_list, lframesnet, rand_trafo, iterations):
    experiments.logger.LOGGER.disabled = True  # turn off logging
    torch.manual_seed(0)
    use_float64 = False

    # create experiment environment
    with hydra.initialize(config_path="../config_quick", version_base=None):
        overrides = [
            *model_list,
            f"model/lframesnet={lframesnet}",
            "save=false",
            f"use_float64={use_float64}",
            # "training.batchsize=1",
        ]
        cfg = hydra.compose(config_name="amplitudes", overrides=overrides)
        exp = AmplitudeExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()

    mse_max = 0
    for i, data in enumerate(exp.train_loader):
        mom = data[1]
        if i == iterations:
            break

        # original data
        amp_original = exp.model(mom)[0]

        # augmented data
        trafo = rand_trafo(mom.shape[:-2] + (1,))
        mom_augmented = torch.einsum(
            "...ij,...j->...i", trafo.to(torch.float64), mom.to(torch.float64)
        ).to(exp.dtype)
        amp_augmented = exp.model(mom_augmented)[0]

        mse = (amp_original - amp_augmented) ** 2
        mse_max = max(mse.max().item(), mse_max)
    print(f"{mse_max:.2e}", model_list, rand_trafo.__name__, lframesnet)
