import torch
import pytest
import hydra

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment
from tensorframes.utils.transforms import rand_rotation, rand_lorentz


@pytest.mark.parametrize("model", ["transformer"])
@pytest.mark.parametrize("lframesnet", ["orthogonal", "polardec"])
@pytest.mark.parametrize("rand_trafo", [rand_rotation, rand_lorentz])
@pytest.mark.parametrize("iterations", [1])
def test_amplitudes(model, lframesnet, rand_trafo, iterations):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../config_quick", version_base=None):
        overrides = [
            f"model={model}",
            f"model/lframesnet={lframesnet}",
            "save=false",
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

        diff = y_pred_augmented - y_pred
        print("Max deviation: ", diff.abs().max())
