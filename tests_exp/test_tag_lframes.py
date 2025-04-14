import torch
import pytest
import hydra
import numpy as np
import os

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment
from experiments.tagging.embedding import embed_tagging_data
from tensorframes.utils.lorentz import lorentz_metric
from tests_exp.utils import crop_particles


@pytest.mark.parametrize(
    "breaking_list",
    [
        [
            "data.beam_reference=null",
            "data.add_time_reference=false",
            "data.add_tagging_features_lframesnet=false",
        ],
    ],
)
@pytest.mark.parametrize(
    "model_idx,model_list",
    list(
        enumerate(
            [
                ["model=tag_ParT"],
                ["model=tag_particlenet-lite"],
                ["model=tag_transformer"],
                ["model=tag_graphnet"],
                ["model=tag_graphnet", "model.include_edges=true"],
            ]
        )
    ),
)
@pytest.mark.parametrize("lframesnet", ["orthogonal", "polardec"])
@pytest.mark.parametrize("operation", ["add", "single"])
@pytest.mark.parametrize("nonlinearity", ["exp"])
@pytest.mark.parametrize("iterations", [1])
@pytest.mark.parametrize("use_float64", [False])
@pytest.mark.parametrize("max_particles", [None, 10])
def test_amplitudes(
    model_idx,
    model_list,
    operation,
    nonlinearity,
    lframesnet,
    breaking_list,
    iterations,
    use_float64,
    max_particles=None,
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
            f"model.lframesnet.equivectors.operation={operation}",
            f"model.lframesnet.equivectors.nonlinearity={nonlinearity}",
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

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    diffs = []
    iterator = iter(cycle(exp.train_loader))
    for _ in range(iterations):
        data = next(iterator)
        data = crop_particles(data, n=max_particles)
        metric = lorentz_metric(data.x.shape[:-1], dtype=exp.dtype)

        embedded_data = embed_tagging_data(
            data.x,
            data.scalars.to(exp.dtype),
            data.ptr,
            exp.cfg.data,
        )

        mom = embedded_data["fourmomenta"]
        global_tagging_features = embedded_data["global_tagging_features"]
        scalars = embedded_data["scalars"]
        ptr = embedded_data["ptr"]
        scalars_withspurions = torch.cat([scalars, global_tagging_features], dim=-1)

        lframes = exp.model.lframesnet(
            fourmomenta=mom, scalars=scalars_withspurions, ptr=ptr
        )
        minkowski_mse = (
            lframes.matrices.transpose(-2, -1) @ metric @ lframes.matrices - metric
        ).pow(2)
        diffs.append(minkowski_mse.detach())

    diffs = torch.cat(diffs, dim=0)
    print(
        f"log-mean={diffs.log().mean().exp():.2e} max={diffs.max().item():.2e}",
        model_list,
        lframesnet,
        operation,
        nonlinearity,
        f"{max_particles=}",
        "float64" if use_float64 else "float32",
    )
    if save:
        os.makedirs("scripts/equi-violation", exist_ok=True)
        filename = (
            f"scripts/equi-violation/equitest_tag_minkowski"
            f">{model_idx}"
            f">{lframesnet}"
            f"~{'float64' if use_float64 else 'float32'}"
            f">{operation}"
            f">{max_particles=}"
            f"~{nonlinearity}.npy"
        )
        np.save(filename, diffs)
