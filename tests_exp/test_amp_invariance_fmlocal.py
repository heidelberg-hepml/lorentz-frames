import torch
import pytest
import hydra
import numpy as np
import os

import experiments.logger
from experiments.amplitudes.experiment import AmplitudeExperiment
from tensorframes.utils.transforms import rand_rotation_uniform, rand_lorentz


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
@pytest.mark.parametrize("operation", ["add", "single"])
@pytest.mark.parametrize("nonlinearity", ["exp"])
@pytest.mark.parametrize("rand_trafo", [rand_rotation_uniform, rand_lorentz])
@pytest.mark.parametrize("iterations", [10])
@pytest.mark.parametrize("use_float64", [False])
def test_amplitudes(
    model_idx,
    model_list,
    lframesnet,
    operation,
    nonlinearity,
    rand_trafo,
    iterations,
    use_float64,
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
            f"use_float64={use_float64}",
            f"model.lframesnet.equivectors.operation={operation}",
            f"model.lframesnet.equivectors.nonlinearity={nonlinearity}",
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
    exp.model.eval()

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    diffs = []
    iterator = iter(cycle(exp.train_loader))
    for _ in range(iterations):
        data = next(iterator)
        mom = data[1]

        parent = super(type(exp.model), exp.model)

        fourmomenta_local = parent.forward(mom)[1]

        # augmented data
        trafo = rand_trafo(mom.shape[:-2] + (1,), dtype=mom.dtype)
        mom_augmented = torch.einsum("...ij,...j->...i", trafo, mom)

        fourmomenta_local_augmented = parent.forward(mom_augmented)[1]

        diff = fourmomenta_local - fourmomenta_local_augmented
        if use_asymmetry:
            diff /= (fourmomenta_local + fourmomenta_local_augmented).clamp(min=1e-20)
        diff = diff**2
        diff[~diff.isfinite()] = 0

        diffs.append(diff.detach())
    diffs = torch.cat(diffs, dim=0)
    print(
        f"log-mean={diffs.log().mean().exp():.2e} max={diffs.max().item():.2e}",
        model_list,
        rand_trafo.__name__,
        lframesnet,
        operation,
        nonlinearity,
        "float64" if use_float64 else "float32",
    )
    if save:
        os.makedirs("scripts/equi-violation", exist_ok=True)
        # ">" for different plots, "~" for different lines in the same plot
        filename = (
            f"scripts/equi-violation/equitest_amp_invariants"
            f">{model_idx}"
            f">{lframesnet}"
            f">{rand_trafo.__name__}"
            f">{'float64' if use_float64 else 'float32'}"
            f">{operation}"
            f"~{nonlinearity}.npy"
        )
        np.save(filename, diffs)
