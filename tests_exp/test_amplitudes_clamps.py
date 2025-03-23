import pytest
import hydra

import experiments.logger
from experiments.amplitudes.experiment import AmplitudeExperiment
from tests_exp.debug import track_clamps


@pytest.mark.parametrize(
    "lframesnet",
    [
        "identity",
        "randomrotation",
        "randomlorentz",
        "orthogonal",
        "polardec",
    ],
)
@pytest.mark.parametrize(
    "model_list",
    [
        ["model=amp_mlp"],
        ["model=amp_transformer"],
        ["model=amp_graphnet"],
        ["model=amp_graphnet", "model.include_edges=false"],
        ["model=amp_graphnet", "model.include_nodes=false"],
    ],
)
@pytest.mark.parametrize("iterations", [100])
def test_amplitudes(lframesnet, model_list, iterations):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../config_quick", version_base=None):
        overrides = [
            *model_list,
            f"model/lframesnet={lframesnet}",
            f"training.iterations={iterations}",
            # "training.batchsize=1",
            "save=false",
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
        with track_clamps() as tracking:
            amp_original = exp.model(mom)[0]

        sorted_calls = sorted(
            tracking,
            key=lambda c: c["num_elements_clamped"] / c["total_elements"],
            reverse=True,
        )

        print("\n")
        for i, c in enumerate(sorted_calls):
            ratio = c["num_elements_clamped"] / c["total_elements"]
            print(
                f"{i+1}. {c['op_type']} clamped {c['num_elements_clamped']} / {c['total_elements']} "
                f"({ratio:.1%}) at {c['filename']}:{c['line']}"
            )
            print(f"   → {c['code']}")
            assert c["num_elements_clamped"] == 0
