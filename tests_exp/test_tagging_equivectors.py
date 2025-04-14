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
from torch_geometric.nn.aggr import MeanAggregation
from tensorframes.lframes.lframes import LFrames


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
                # ["model=tag_particlenet-lite"],
                ["model=tag_transformer"],
                # ["model=tag_graphnet"],
                # ["model=tag_graphnet", "model.include_edges=false"],
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
    ],  # , "relu", "relu_shifted", "top10_softplus", "top10_softmax"]
)
@pytest.mark.parametrize("iterations", [10])
@pytest.mark.parametrize("use_float64", [False, True])
def test_amplitudes(
    rand_trafo,
    model_idx,
    model_list,
    operation,
    nonlinearity,
    lframesnet,
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

    mses = []
    infs = []
    agg = MeanAggregation()
    iterator = iter(cycle(exp.train_loader))
    for _ in range(iterations):
        data = next(iterator).to(exp.device)
        data_augmented = data.clone().to(exp.device)

        embedded_data = embed_tagging_data(
            data.x.to(exp.dtype),
            data.scalars.to(exp.dtype),
            data.ptr,
            exp.cfg.data,
        )

        fourmomenta = embedded_data["fourmomenta"]
        global_tagging_features = embedded_data["global_tagging_features"]
        scalars = embedded_data["scalars"]
        ptr = embedded_data["ptr"]
        batch_nospurions = embedded_data["batch"]

        assert fourmomenta.shape[0] == data.x.shape[0]  # no spurions
        scalars_withspurions = torch.cat([scalars, global_tagging_features], dim=-1)
        vecs = exp.model.lframesnet.equivectors(
            fourmomenta,
            scalars=scalars_withspurions,
            ptr=ptr,
        )

        # augmented data
        mom = data_augmented.x.to(exp.device)
        trafo = rand_trafo(mom.shape[:-2] + (1,), dtype=mom.dtype)
        mom_augmented = torch.einsum("...ij,...j->...i", trafo, mom)
        data_augmented.x = mom_augmented

        embedded_data_augmented = embed_tagging_data(
            data_augmented.x.to(exp.dtype),
            data_augmented.scalars.to(exp.dtype),
            data_augmented.ptr,
            exp.cfg.data,
        )

        fourmomenta_augmented = embedded_data_augmented["fourmomenta"]
        global_tagging_features_augmented = embedded_data_augmented[
            "global_tagging_features"
        ]
        scalars_augmented = embedded_data_augmented["scalars"]
        scalars_withspurions_augmented = torch.cat(
            [scalars_augmented, global_tagging_features_augmented], dim=-1
        )
        assert fourmomenta_augmented.shape[0] == data.x.shape[0]  # no spurions
        vecs_augmented = exp.model.lframesnet.equivectors(
            fourmomenta_augmented,
            scalars=scalars_withspurions_augmented,
            ptr=ptr,
        )

        lframes = LFrames(trafo.to(exp.dtype))
        vecs_augmented = torch.einsum(
            "...ij,...j->...i", lframes.inv, vecs_augmented.to(exp.dtype)
        ).to(exp.dtype)

        norm = vecs + vecs_augmented
        infs.append(torch.any(torch.isinf(norm)).item())

        mse = ((vecs - vecs_augmented) / norm).pow(2).mean(dim=-2)
        mses.append(agg(mse, index=batch_nospurions))

    mses = torch.cat(mses, dim=0)
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
        filename = (
            f"scripts/equi-violation/equitest_tag_equivectors"
            f">{model_idx}"
            f">{lframesnet}"
            f">{rand_trafo.__name__}"
            f">{'float64' if use_float64 else 'float32'}"
            f">{operation}"
            f"~{nonlinearity}.npy"
        )
        np.save(filename, mses.detach().cpu().numpy())
