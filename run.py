import hydra
from experiments.toptagging.experiment import TopTaggingExperiment


@hydra.main(config_path="config", config_name="toptagging", version_base=None)
def main(cfg):
    if cfg.exp_type == "toptagging":
        exp = TopTaggingExperiment(cfg)
    else:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    exp()


if __name__ == "__main__":
    main()
