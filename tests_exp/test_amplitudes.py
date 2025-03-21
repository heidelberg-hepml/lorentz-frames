import torch
import pytest
import hydra

import experiments.logger
from experiments.amplitudes.experiment import AmplitudeExperiment
from tensorframes.utils.transforms import rand_rotation, rand_lorentz


@pytest.mark.parametrize("model", ["amp_transformer"])
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
        cfg = hydra.compose(config_name="amplitudes", overrides=overrides)
        exp = AmplitudeExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()

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
        ).to(torch.float32)
        amp_augmented = exp.model(mom_augmented)[0]

        diff = amp_original - amp_augmented
        print("Max deviation: ", diff.abs().max())
