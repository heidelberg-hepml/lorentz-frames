#!/usr/bin/env python3

import hydra
import optuna
import torch
from hydra import compose, initialize
from optuna import Trial
from pathlib import Path

from experiments.tagging.experiment import TopTaggingExperiment


def run_trial(trial: Trial, seed, exp_name, model, cfg_overrides):
    """Performs a trial: samples hyperparameters, trains a model, and returns the validation error"""

    # Choose trial params
    # global parameters
    lframesnet = trial.suggest_categorical("lframesnet", ["orthogonal", "polardec"])
    # experiment
    mass_reg = 10 ** trial.suggest_int("log10_mass_reg", -10, 0)
    spurion_scale = trial.suggest_categorical(
        "spurion_scale", [1e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 1e2]
    )
    add_tagging_features_lframesnet = trial.suggest_categorical(
        "add_tagging_features_lframesnet", ["true", "false"]
    )
    beam_reference = trial.suggest_categorical(
        "beam_reference", ["lightlike", "timelike"]
    )
    two_beams = trial.suggest_categorical("two_beams", ["true", "false"])
    # lframesnet
    eps_norm = 10 ** trial.suggest_int("log10_eps_norm", -20, -2)
    eps_reg_coplanar = 10 ** trial.suggest_int("log10_eps_reg_coplanar", -20, -2)
    eps_reg_lightlike = 10 ** trial.suggest_int("log10_eps_reg_lightlike", -20, -2)
    eps_reg = 10 ** trial.suggest_int("log10_eps_reg", -20, -2)
    # equigraph
    hidden_channels = 2 ** trial.suggest_int("log2_hidden_channels", 4, 7)
    hidden_layers_mlp = trial.suggest_int("hidden_layers_mlp", 1, 4)
    nonlinearity = trial.suggest_categorical(
        "nonlinearity",
        [
            "null",
            "exp",
            "softplus",
            "softmax",
            "relu_shifted",
            "top5_exp",
            "top10_exp",
            "top20_exp",
        ],
    )
    dropout_lframesnet = trial.suggest_categorical(
        "dropout_lframesnet", [0, 0.01, 0.3, 0.1, 0.2, 0.3]
    )
    # training
    lr_factor_lframesnet = trial.suggest_categorical(
        "lr_factor_lframesnet", [0.1, 0.3, 1.0, 3.0, 10.0]
    )
    weight_decay_lframesnet = trial.suggest_categorical(
        "weight_decay_lframesnet", [0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 0.2]
    )

    # catch runs with invalid hyperparameters
    if False:
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
            f"data.spurion_scale={spurion_scale}",
            f"data.add_tagging_features_lframesnet={add_tagging_features_lframesnet}",
            f"data.beam_reference={beam_reference}",
            f"data.two_beams={two_beams}",
            f"model.lframesnet.ortho_kwargs.eps_norm={eps_norm}",
            f"model.lframesnet.equivectors.hidden_channels={hidden_channels}",
            f"model.lframesnet.equivectors.num_layers_mlp={hidden_layers_mlp}",
            f"model.lframesnet.equivectors.nonlinearity={nonlinearity}",
            f"model.lframesnet.equivectors.dropout_prob={dropout_lframesnet}",
            f"training.weight_decay_lframesnet={weight_decay_lframesnet}",
            f"training.lr_factor_lframesnet={lr_factor_lframesnet}",
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

        cfg = compose(config_name="toptagging", overrides=overrides)
        try:
            exp = TopTaggingExperiment(cfg)
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
        score = exp.results["val"]["auc"]  # use validation AUC as score

        return score


@hydra.main(config_path=".", config_name="sweep-top1", version_base=None)
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
