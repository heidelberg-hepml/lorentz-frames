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
from experiments.tagging.embedding import embed_tagging_data
from tests_exp.utils import crop_particles


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
    rand_trafo,
    model_idx,
    model_list,
    lframesnet,
    operation,
    nonlinearity,
    breaking_list,
    iterations,
    use_float64,
    max_particles,
    use_asymmetry=True,
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
    exp.model.eval()

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    diffs = []
    iterator = iter(cycle(exp.train_loader))
    for _ in range(iterations):
        data = next(iterator).to(exp.device)
        data = crop_particles(data, n=max_particles)
        data_augmented = data.clone().to(exp.device)

        parent = super(type(exp.model), exp.model)
        embedded_data = embed_tagging_data(
            data.x,
            data.scalars.to(exp.dtype),
            data.ptr,
            exp.cfg.data,
        )
        fourmomenta_local_nospurions = parent.forward(embedded_data)[1]

        # augmented data
        mom = data_augmented.x
        trafo = rand_trafo(mom.shape[:-2] + (1,), dtype=mom.dtype)
        mom_augmented = torch.einsum("...ij,...j->...i", trafo.to(exp.device), mom)
        data_augmented.x = mom_augmented

        embedded_data_augmented = embed_tagging_data(
            data_augmented.x,
            data_augmented.scalars.to(exp.dtype),
            data_augmented.ptr,
            exp.cfg.data,
        )
        fourmomenta_local_nospurions_augmented = parent.forward(
            embedded_data_augmented
        )[1]

        diff = fourmomenta_local_nospurions - fourmomenta_local_nospurions_augmented
        if use_asymmetry:
            diff /= (
                fourmomenta_local_nospurions + fourmomenta_local_nospurions_augmented
            ).abs().clamp(min=1e-20)
        diff = diff**2
        diff[~diff.isfinite()] = 0

        diffs.append(diff.detach())
    diffs = torch.cat(diffs, dim=0).clamp(min=1e-20)
    print(
        f"log-mean={diffs.log().mean().exp():.2e} max={diffs.max().item():.2e}",
        model_list,
        rand_trafo.__name__,
        lframesnet,
        operation,
        nonlinearity,
        f"{max_particles=}",
        "float64" if use_float64 else "float32",
    )
    if save:
        os.makedirs("scripts/invariance_fmlocal", exist_ok=True)
        filename = (
            f"scripts/invariance_fmlocal/equitest_tag_invariants"
            f">{model_idx}"
            f">{lframesnet}"
            f"~{rand_trafo.__name__}"
            f"~{'float64' if use_float64 else 'float32'}"
            f">{operation}"
            f"~{max_particles}"
            f">{nonlinearity}.npy"
        )
        np.save(filename, diffs.detach().cpu())
