import hydra
from experiments.tagging.experiment import TopTaggingExperiment
from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment


@hydra.main(config_path="config_quick", config_name="toptagging", version_base=None)
def main(cfg):
    if cfg.exp_type == "toptagging":
        exp = TopTaggingExperiment(cfg)
    elif cfg.exp_type == "jctagging":
        exp = JetClassTaggingExperiment(cfg)
    else:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    exp()


if __name__ == "__main__":
    main()
