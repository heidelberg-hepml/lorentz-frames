import os, torch
from omegaconf import OmegaConf, open_dict
from torch_ema import ExponentialMovingAverage

from experiments.tagging.experiment import TopTaggingExperiment
from experiments.logger import LOGGER


class TopTaggingFineTuneExperiment(TopTaggingExperiment):
    def __init__(self, cfg):
        super().__init__(cfg)

        # load warm_start cfg
        warmstart_path = os.path.join(
            self.cfg.finetune.backbone_path, self.cfg.finetune.backbone_cfg
        )
        self.warmstart_cfg = OmegaConf.load(warmstart_path)
        assert self.warmstart_cfg.exp_type in ["jctagging"]  # could extend to topxl
        assert self.warmstart_cfg.data.features == "fourmomenta"

        if not self.warmstart_cfg.model._target_ in [
            "experiments.tagging.wrappers.TransformerWrapper",
        ]:
            raise NotImplementedError

        # merge config files
        with open_dict(self.cfg):
            # overwrite model
            self.cfg.model = self.warmstart_cfg.model
            self.cfg.ema = self.warmstart_cfg.ema

            # overwrite model-specific cfg.data entries
            # NOTE: might have to extend this if adding more models
            self.cfg.data.add_tagging_features_lframesnet = (
                self.warmstart_cfg.data.add_tagging_features_lframesnet
            )
            self.cfg.data.beam_reference = self.warmstart_cfg.data.beam_reference
            self.cfg.data.two_beams = self.warmstart_cfg.data.two_beams
            self.cfg.data.add_time_reference = (
                self.warmstart_cfg.data.add_time_reference
            )
            self.cfg.data.mass_reg = self.warmstart_cfg.data.mass_reg
            self.cfg.data.spurion_scale = self.warmstart_cfg.data.spurion_scale
            self.cfg.data.momentum_float64 = self.warmstart_cfg.data.momentum_float64

    def init_model(self):
        # overwrite output channel shape to allow loading pretrained weights
        self.cfg.model.out_channels = self.warmstart_cfg.model.out_channels

        super().init_model()

        if self.warm_start:
            # nothing to do
            return

        # load pretrained weights
        model_path = os.path.join(
            self.warmstart_cfg.run_dir,
            "models",
            f"model_run{self.warmstart_cfg.run_idx}.pt",
        )
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)[
                "model"
            ]
        except FileNotFoundError:
            raise ValueError(f"Cannot load model from {model_path}")
        LOGGER.info(f"Loading pretrained model from {model_path}")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device, dtype=self.dtype)

        # overwrite output layer
        if (
            self.warmstart_cfg.model._target_
            == "experiments.tagging.wrappers.TransformerWrapper"
        ):
            self.model.net.linear_out = torch.nn.Linear(
                self.model.net.hidden_channels, self.num_outputs
            ).to(self.device)
        else:
            raise NotImplementedError

        if self.cfg.ema:
            LOGGER.info(f"Re-initializing EMA")
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=self.cfg.training.ema_decay
            ).to(self.device)

    def _init_optimizer(self):
        # collect parameter lists
        if (
            self.warmstart_cfg.model._target_
            == "experiments.tagging.wrappers.TransformerWrapper"
        ):
            params_backbone_lfnet = list(self.model.lframesnet.parameters())
            params_backbone_main = list(self.model.net.linear_in.parameters()) + list(
                self.model.net.blocks.parameters()
            )
            params_head = self.model.net.linear_out.parameters()

            # assign parameter-specific learning rates
            param_groups = [
                {
                    "params": params_backbone_lfnet,
                    "lr": self.cfg.finetune.lr_backbone
                    * self.cfg.training.lr_factor_lframesnet,
                    "weight_decay": self.cfg.training.weight_decay_lframesnet,
                },
                {
                    "params": params_backbone_main,
                    "lr": self.cfg.finetune.lr_backbone,
                    "weight_decay": self.cfg.training.weight_decay,
                },
                {
                    "params": params_head,
                    "lr": self.cfg.finetune.lr_head,
                    "weight_decay": self.cfg.training.weight_decay,
                },
            ]
        else:
            raise NotImplementedError

        super()._init_optimizer(param_groups=param_groups)
