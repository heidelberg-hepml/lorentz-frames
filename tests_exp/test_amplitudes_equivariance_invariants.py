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
                # ["model=amp_transformer"],
                # ["model=amp_graphnet"],
                # ["model=amp_graphnet", "model.include_edges=false"],
                # ["model=amp_graphnet", "model.include_nodes=false"],
            ],
        )
    ),
)
@pytest.mark.parametrize("lframesnet", ["orthogonal", "polardec"])
@pytest.mark.parametrize("operation", ["add", "single"])
@pytest.mark.parametrize(
    "nonlinearity", ["softplus", "top5_softplus", "top10_softplus", "top20_softplus"]
)
@pytest.mark.parametrize("rand_trafo", [rand_rotation_uniform, rand_lorentz])
@pytest.mark.parametrize("iterations", [10])
@pytest.mark.parametrize("use_float64", [False, True])
def test_amplitudes(
    model_idx,
    model_list,
    lframesnet,
    operation,
    nonlinearity,
    rand_trafo,
    iterations,
    use_float64,
    save=True,
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

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    mses = []
    infs = []
    iterator = iter(cycle(exp.train_loader))
    for _ in range(iterations):
        data = next(iterator)
        mom = data[1].to(exp.device)

        parent = super(type(exp.model), exp.model)

        (
            features_local,
            fourmomenta_local,
            particle_type,
            lframes,
            tracker,
        ) = parent.forward(mom)

        # augmented data
        trafo = rand_trafo(mom.shape[:-2] + (1,)).to(exp.device)
        mom_augmented = torch.einsum(
            "...ij,...j->...i", trafo.to(torch.float64), mom.to(torch.float64)
        ).to(exp.dtype)

        (
            features_local_augmented,
            fourmomenta_local_augmented,
            particle_type_augmented,
            lframes_augmented,
            tracker,
        ) = parent.forward(mom_augmented)

        norm = fourmomenta_local + fourmomenta_local_augmented
        diff = ((fourmomenta_local - fourmomenta_local_augmented) / norm) ** 2
        infs.append((~diff.isfinite()).sum().detach().item())
        diff[~diff.isfinite()] = 1e-12
        diff = diff.mean(dim=-2)

        mses.append(diff.detach())
    mses = torch.cat(mses, dim=0).clamp(min=1e-30)
    # print("infs: ", infs)
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
        np.save(filename, mses.cpu().numpy())
