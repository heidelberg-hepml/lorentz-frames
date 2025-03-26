import torch
import pytest
import hydra

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment
from tensorframes.utils.transforms import rand_rotation, rand_lorentz, rand_xyrotation


@pytest.mark.parametrize(
    "rand_trafo,breaking_list",
    [
        [rand_rotation, ["data.beam_reference=null", "data.add_time_reference=false"]],
        [rand_lorentz, ["data.beam_reference=null", "data.add_time_reference=false"]],
        [rand_xyrotation, []],
    ],
)
@pytest.mark.parametrize(
    "model_list",
    [
        ["model=transformer"],
        ["model=graphnet"],
        ["model=graphnet", "model.include_edges=false"],
    ],
)
@pytest.mark.parametrize("lframesnet", ["orthogonal", "polardec"])
@pytest.mark.parametrize("iterations", [1])
def test_amplitudes(rand_trafo, model_list, lframesnet, breaking_list, iterations):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../config_quick", version_base=None):
        overrides = [
            *model_list,
            f"model/lframesnet={lframesnet}",
            "save=false",
            *breaking_list,
            # "training.batchsize=1",
        ]
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
        exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()

    mse_max = 0
    for i, data in enumerate(exp.train_loader):
        data_augmented = data.clone()
        if i == iterations:
            break

        # original data
        y_pred = exp._get_ypred_and_label(data)[0]

        # augmented data
        mom = data_augmented.x
        trafo = rand_trafo(mom.shape[:-2] + (1,))
        mom_augmented = torch.einsum(
            "...ij,...j->...i", trafo.to(torch.float64), mom.to(torch.float64)
        ).to(torch.float32)
        data_augmented.x = mom_augmented
        y_pred_augmented = exp._get_ypred_and_label(data_augmented)[0]

        mse = (y_pred_augmented - y_pred) ** 2
        mse_max = max(mse.max().item(), mse_max)
    print(f"{mse_max:.2e}", rand_trafo.__name__, lframesnet, breaking_list, model_list)
