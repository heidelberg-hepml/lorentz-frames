import torch
import pytest
import hydra
import numpy as np

import experiments.logger
from experiments.amplitudes.experiment import AmplitudeExperiment
from tensorframes.utils.transforms import rand_rotation, rand_lorentz


@pytest.mark.parametrize(
    "model_idx,model_list",
    list(
        enumerate(
            [
                ["model=amp_mlp"],
                ["model=amp_transformer"],
                ["model=amp_graphnet"],
                ["model=amp_graphnet", "model.include_edges=false"],
                ["model=amp_graphnet", "model.include_nodes=false"],
            ],
        )
    ),
)
@pytest.mark.parametrize("lframesnet", ["orthogonal", "polardec"])
@pytest.mark.parametrize("rand_trafo", [rand_rotation, rand_lorentz])
@pytest.mark.parametrize("iterations", [100])
@pytest.mark.parametrize("use_float64", [False, True])
def test_amplitudes(
    model_idx, model_list, lframesnet, rand_trafo, iterations, use_float64
):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../config_quick", version_base=None):
        overrides = [
            *model_list,
            f"model/lframesnet={lframesnet}",
            "save=false",
            # f"use_float64={use_float64}",
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

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    mses = []
    iterator = iter(cycle(exp.train_loader))
    for _ in range(iterations):
        data = next(iterator)
        mom = data[1]

        # original data
        amp_original = exp.model(mom)[0]

        # augmented data
        trafo = rand_trafo(mom.shape[:-2] + (1,))
        mom_augmented = torch.einsum(
            "...ij,...j->...i", trafo.to(torch.float64), mom.to(torch.float64)
        ).to(exp.dtype)
        amp_augmented = exp.model(mom_augmented)[0]

        mse = (amp_original - amp_augmented) ** 2
        mses.append(mse.detach())
    mses = torch.cat(mses, dim=0)[:, 0].clamp(min=1e-30)
    print(
        f"log-mean={mses.log().mean().exp():.2e} max={mses.max().item():.2e}",
        model_list,
        rand_trafo.__name__,
        lframesnet,
    )
    filename = (
        f"tests_exp/equitest_amp_{model_idx}_{lframesnet}_{rand_trafo.__name__}.npy"
    )
    np.save(filename, mses.cpu().numpy())
