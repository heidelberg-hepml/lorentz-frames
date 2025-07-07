#!/usr/bin/env python3

import hydra
import optuna
import torch
from hydra import compose, initialize
from optuna import Trial
from pathlib import Path

from experiments.amplitudes.experimentxl import AmplitudeXLExperiment


def run_trial(trial: Trial, seed, exp_name, model, cfg_overrides):
    """Performs a trial: samples hyperparameters, trains a model, and returns the validation error"""

    # Choose trial params
    # global parameters
    lframesnet = trial.suggest_categorical("lframesnet", ["orthogonal", "polardec"])
    # experiment
    mass_reg = 10 ** trial.suggest_int("log10_mass_reg", -10, 2)
    # lframesnet
    method = trial.suggest_categorical("method", ["gramschmidt", "cross"])
    eps_norm = 10 ** trial.suggest_int("log10_eps_norm", -20, -2)
    eps_reg_coplanar = 10 ** trial.suggest_int("log10_eps_reg_coplanar", -20, -2)
    eps_reg_lightlike = 10 ** trial.suggest_int("log10_eps_reg_lightlike", -20, -2)
    eps_reg = 10 ** trial.suggest_int("log10_eps_reg", -20, -2)
    # equigraph
    hidden_channels = 2 ** trial.suggest_int("log2_hidden_channels", 4, 7)
    hidden_layers_mlp = trial.suggest_int("hidden_layers_mlp", 1, 4)
    operation = trial.suggest_categorical("operation", ["single", "add"])
    nonlinearity = trial.suggest_categorical(
        "nonlinearity",
        [
            "null",
            "exp",
            "softplus",
            "softmax",
            "relu_shifted",
            "top3_exp",
            "top3_softplus",
        ],
    )
    layer_norm = trial.suggest_categorical("layer_norm", ["true", "false"])
    fm_norm = trial.suggest_categorical("fm_norm", ["true", "false"])

    # catch runs with invalid hyperparameters
    if operation == "single" and fm_norm == "true":
        raise optuna.TrialPruned()

    with initialize(config_path="../config", version_base=None):
        overrides = cfg_overrides
        overrides += [
            # optuna-related settings
            f"run_name=trial_{trial.number}",
            f"exp_name={exp_name}",
            f"seed={seed}",
            # Fixed parameters
            f"model={model}",
            f"training={model}",
            # Tuned parameters
            f"model/lframesnet={lframesnet}",
            f"data.mass_reg={mass_reg}",
            f"model.lframesnet.ortho_kwargs.method={method}",
            f"model.lframesnet.ortho_kwargs.eps_norm={eps_norm}",
            f"model.lframesnet.equivectors.hidden_channels={hidden_channels}",
            f"model.lframesnet.equivectors.num_layers_mlp={hidden_layers_mlp}",
            f"model.lframesnet.equivectors.operation={operation}",
            f"model.lframesnet.equivectors.nonlinearity={nonlinearity}",
            f"model.lframesnet.equivectors.layer_norm={layer_norm}",
            f"model.lframesnet.equivectors.fm_norm={fm_norm}",
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

        cfg = compose(config_name="amplitudesxl", overrides=overrides)
        try:
            exp = AmplitudeXLExperiment(cfg)
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            print("Pruning trial {trial.number} due to torch.cuda.OutOfMemoryError")
            raise optuna.TrialPruned()
        except AssertionError as e:
            print(e)
            print("Pruning trial {trial.number} due to AssertionError")
            raise optuna.TrialPruned()
        except RuntimeError as e:
            print(e)
            print("Pruning trial {trial.number} due to RuntimeError")
            raise optuna.TrialPruned()

        # Run experiment
        exp()
        score = exp.results["val"]["prepd"]["mse"]  # use validation MSE as score

        return score


@hydra.main(config_path=".", config_name="sweep-amp1", version_base=None)
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
