#!/usr/bin/env python3

import hydra
import optuna
import numpy as np
from hydra import compose, initialize
from optuna import Trial
from pathlib import Path

from experiments.eventgen.processes import ttbarExperiment


def run_trial(trial: Trial, seed, exp_name, model, cfg_overrides):
    """Performs a trial: samples hyperparameters, trains a model, and returns the validation error"""

    # Choose trial params
    # global parameters
    lframesnet = trial.suggest_categorical("lframesnet", ["orthogonal", "polardec"])
    # experiment
    beam_reference = trial.suggest_categorical(
        "beam_reference", ["lightlike", "timelike"]
    )
    two_beams = trial.suggest_categorical("two_beams", ["true", "false"])
    # lframesnet
    eps_norm = 10 ** trial.suggest_int("log10_eps_norm", -15, -4)
    eps_reg_coplanar = 10 ** trial.suggest_int("log10_eps_reg_coplanar", -15, -4)
    eps_reg_lightlike = 10 ** trial.suggest_int("log10_eps_reg_lightlike", -15, -4)
    eps_reg = 10 ** trial.suggest_int("log10_eps_reg", -15, -4)
    # equigraph
    hidden_channels = 2 ** trial.suggest_int("log2_hidden_channels", 4, 7)
    hidden_layers_mlp = trial.suggest_int("hidden_layers_mlp", 1, 4)
    nonlinearity = trial.suggest_categorical(
        "nonlinearity",
        [
            "exp",
            "softplus",
            "softmax",
        ],
    )
    # training
    lr_factor_lframesnet = trial.suggest_categorical(
        "lr_factor_lframesnet", [0.1, 0.3, 1, 3, 10]
    )
    clip_grad_norm = trial.suggest_float("clip_grad_norm", 0.1, 10.0, log=True)

    # catch runs with invalid hyperparameters
    if False:
        raise optuna.TrialPruned()

    with initialize(config_path="../config", version_base=None):
        overrides = cfg_overrides
        overrides += [
            # optuna-related settings
            f"run_name=trial_{trial.number}",
            f"exp_name={exp_name}",
            # f"seed={seed}",
            # Fixed parameters
            f"model={model}",
            # Tuned parameters
            f"model/lframesnet={lframesnet}",
            f"data.spurions.beam_reference={beam_reference}",
            f"data.spurions.two_beams={two_beams}",
            f"model.lframesnet.ortho_kwargs.eps_norm={eps_norm}",
            f"model.lframesnet.equivectors.hidden_channels={hidden_channels}",
            f"model.lframesnet.equivectors.num_layers_mlp={hidden_layers_mlp}",
            f"model.lframesnet.equivectors.nonlinearity={nonlinearity}",
            f"model.lframesnet.ortho_kwargs.eps_norm={eps_norm}",
            f"training.lr_factor_lframesnet={lr_factor_lframesnet}",
            f"training.clip_grad_norm={clip_grad_norm}",
        ]
        if lframesnet == "orthogonal":
            overrides += [
                f"model.lframesnet.ortho_kwargs.eps_reg_coplanar={eps_reg_coplanar}",
                f"model.lframesnet.ortho_kwargs.eps_reg_lightlike={eps_reg_lightlike}",
            ]
        elif lframesnet == "polardec":
            overrides += [
                f"model.lframesnet.ortho_kwargs.eps_reg={eps_reg}",
            ]

        cfg = compose(config_name="ttbar", overrides=overrides)
        try:
            exp = ttbarExperiment(cfg)
        except Exception as e:
            # raise exception -> dont count it towards startup steps
            raise

        # Run experiment
        exp()
        score = np.mean(exp.NLLs)  # use validation AUC as score

        if score < -33:
            print("Flagging trial {trial.number} due to low score NLL={score}")
            raise ValueError(f"Not-trustworthy run")

        return score


@hydra.main(config_path=".", config_name="sweep-eg1", version_base=None)
def sweep(cfg):
    """Entrance point to parameter sweep (wrapped with hydra)"""

    # Clear hydra instances
    # Important so we can use hydra again in the experiment
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    # Create or load study
    Path(cfg.optuna_db).parent.mkdir(exist_ok=True, parents=True)
    study = optuna.create_study(
        storage=f"sqlite:///{Path(cfg.optuna_db).resolve()}?timeout=60",
        load_if_exists=True,
        study_name=cfg.exp_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            multivariate=True, n_startup_trials=cfg.n_startup_trials
        ),
    )

    # Let's go
    study.optimize(
        lambda trial: run_trial(
            trial,
            seed=cfg.seed,
            exp_name=cfg.exp_name,
            model=cfg.model_name,
            cfg_overrides=cfg.cfg_overrides,
        ),
        n_trials=cfg.trials,
    )


# execute with 'python -m sweep.sweep-amp seed=0' (or any other hydra overrides)
if __name__ == "__main__":
    sweep()
