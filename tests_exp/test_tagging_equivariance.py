import torch
import pytest
import hydra
import numpy as np
import os

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment
from tensorframes.utils.transforms import (
    rand_rotation_uniform,
    rand_lorentz,
    rand_xyrotation,
)


BREAKING = [
    "data.beam_reference=null",
    "data.add_time_reference=false",
    "data.add_tagging_features_lframesnet=false",
]


@pytest.mark.parametrize(
    "rand_trafo,breaking_list",
    [
        [rand_rotation_uniform, BREAKING],
        [rand_lorentz, BREAKING],
        [rand_xyrotation, BREAKING],
    ],
)
@pytest.mark.parametrize(
    "model_idx,model_list",
    list(
        enumerate(
            [
                ["model=tag_particlenet-lite"],
                ["model=tag_transformer"],
                ["model=tag_graphnet"],
                ["model=tag_graphnet", "model.include_edges=false"],
            ]
        )
    ),
)
@pytest.mark.parametrize("lframesnet", ["orthogonal", "polardec"])
@pytest.mark.parametrize("operation", ["add", "single"])
@pytest.mark.parametrize(
    "nonlinearity",
    [
        "exp",
        "softplus",
        "softmax",
        "relu",
        "relu_shifted",
        "top10_softplus",
        "top10_softmax",
    ],
)
@pytest.mark.parametrize("iterations", [100])
@pytest.mark.parametrize("use_float64", [False, True])
def test_amplitudes(
    rand_trafo,
    model_idx,
    model_list,
    lframesnet,
    operation,
    nonlinearity,
    breaking_list,
    iterations,
    use_float64,
    save=False,
):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../config_quick", version_base=None):
        overrides = [
            *model_list,
            f"model/lframesnet={lframesnet}",
            "save=false",
            *breaking_list,
            f"use_float64={use_float64}",
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
    exp.model.eval()  # turn off dropout

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    mses = []
    iterator = iter(cycle(exp.train_loader))
    for _ in range(iterations):
        data = next(iterator)
        data_augmented = data.clone()

        # original data
        y_pred = exp._get_ypred_and_label(data)[0]

        # augmented data
        mom = data_augmented.x
        trafo = rand_trafo(mom.shape[:-2] + (1,), dtype=mom.dtype)
        mom_augmented = torch.einsum("...ij,...j->...i", trafo, mom)
        data_augmented.x = mom_augmented
        y_pred_augmented = exp._get_ypred_and_label(data_augmented)[0]

        mse = (y_pred_augmented - y_pred) ** 2
        mses.append(mse.detach())
    mses = torch.cat(mses, dim=0).clamp(min=1e-30)
    print(
        f"log-mean={mses.log().mean().exp():.2e} max={mses.max().item():.2e}",
        model_list,
        rand_trafo.__name__,
        lframesnet,
        operation,
        nonlinearity,
        "float64" if use_float64 else "float32",
    )
    if save:
        os.makedirs("scripts/equi-violation", exist_ok=True)
        filename = (
            f"scripts/equi-violation/equitest_tag"
            f">{model_idx}"
            f">{lframesnet}"
            f">{rand_trafo.__name__}"
            f">{'float64' if use_float64 else 'float32'}"
            f">{operation}"
            f"~{nonlinearity}.npy"
        )
        np.save(filename, mses.cpu().numpy())
