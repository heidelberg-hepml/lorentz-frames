import torch
import pytest
import hydra
import numpy as np
import os

import experiments.logger
from experiments.amplitudes.experiment import AmplitudeExperiment
from tensorframes.utils.transforms import rand_rotation_uniform, rand_lorentz
from tensorframes.lframes.lframes import LFrames


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
@pytest.mark.parametrize("iterations", [100])
@pytest.mark.parametrize("use_float64", [False])
def test_amplitudes(
    model_idx,
    model_list,
    operation,
    nonlinearity,
    lframesnet,
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

        particle_type = exp.model.encode_particle_type(mom.shape[0]).to(
            dtype=exp.model.input_dtype
        )
        vecs = exp.model.lframesnet.equivectors(mom, scalars=particle_type, ptr=None)

        # augmented data
        trafo = rand_trafo(mom.shape[:-2], dtype=mom.dtype)
        mom_augmented = torch.einsum("...nm,...vm->...vn", trafo, mom)

        particle_type_augmented = exp.model.encode_particle_type(
            mom_augmented.shape[0]
        ).to(dtype=exp.model.input_dtype)
        vecs_augmented = exp.model.lframesnet.equivectors(
            mom_augmented, scalars=particle_type_augmented, ptr=None
        )

        lframes = LFrames(trafo)
        vecs_augmented = torch.einsum(
            "...nm,...vim->...vin", lframes.inv, vecs_augmented
        )

        diff = vecs - vecs_augmented
        if use_asymmetry:
            diff /= (vecs + vecs_augmented).abs().clamp(min=1e-20)
        diff = diff**2
        diffs.append(diff)
    diffs = torch.cat(diffs, dim=0).clamp(min=1e-20)
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
            f"scripts/equi-violation/equitest_amp_equivectors"
            f">{model_idx}"
            f">{lframesnet}"
            f">{rand_trafo.__name__}"
            f">{'float64' if use_float64 else 'float32'}"
            f">{operation}"
            f"~{nonlinearity}.npy"
        )
        np.save(filename, diffs)
