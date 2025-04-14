import torch
from torch.utils.data import DataLoader

import os, time

from experiments.logger import LOGGER

from experiments.tagging.experiment import TaggingExperiment
from experiments.tagging.embedding import (
    dense_to_sparse_jet,
    embed_tagging_data,
)

from experiments.tagging.miniweaver.dataset import SimpleIterDataset
from experiments.tagging.miniweaver.loader import to_filelist


class TopXLTaggingExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.cfg.data.features == "fourmomenta":
            self.cfg.data.data_config = (
                "experiments/tagging/miniweaver/configs_topxl/fourmomenta.yaml"
            )
        elif self.cfg.data.features == "pid":
            self.cfg.model.in_channels += 6
            self.cfg.data.data_config = (
                "experiments/tagging/miniweaver/configs_topxl/pid.yaml"
            )
        elif self.cfg.data.features == "displacements":
            self.cfg.model.in_channels += 4
            self.cfg.data.data_config = (
                "experiments/tagging/miniweaver/configs_topxl/displacements.yaml"
            )
        elif self.cfg.data.features == "default":
            self.cfg.model.in_channels += 10
            self.cfg.data.data_config = (
                "experiments/tagging/miniweaver/configs_topxl/default.yaml"
            )
        else:
            raise ValueError(
                f"Input feature option {self.cfg.data.features} not implemented"
            )

    def init_physics(self):
        # decide which entries to use for the lframesnet
        if "equivectors" in self.cfg.model.lframesnet:
            self.cfg.model.lframesnet.equivectors.num_scalars = (
                self.cfg.model.in_channels
                if self.cfg.data.add_tagging_features_lframesnet
                else self.cfg.model.in_channels - 7
            )

        if self.cfg.model.net._target_.rsplit(".", 1)[-1] == "TFGraphNet":
            self.cfg.model.net.num_edge_attr = 1 if self.cfg.model.include_edges else 0

    def init_data(self):
        LOGGER.info("Creating SimpleIterDataset")
        t0 = time.time()

        datasets = {"train": None, "test": None, "val": None}

        for_training = {"train": True, "val": True, "test": False}
        folder = {"train": "train_topxl", "test": "test_topxl", "val": "val_topxl"}
        files_range = {
            "train": self.cfg.data.train_files_range,
            "test": self.cfg.data.test_files_range,
            "val": self.cfg.data.val_files_range,
        }
        self.num_files = {
            label: frange[1] - frange[0] for label, frange in files_range.items()
        }
        for label in ["train", "test", "val"]:
            path = os.path.join(self.cfg.data.data_dir, folder[label])
            flist = [
                f"{path}/file_{str(i).zfill(3)}.parquet"
                for i in range(*files_range[label])
            ]
            file_dict, _ = to_filelist(flist)

            LOGGER.info(f"Using {len(flist)} files for {label}ing from {path}")
            datasets[label] = SimpleIterDataset(
                file_dict,
                self.cfg.data.data_config,
                for_training=for_training[label],
                extra_selection=self.cfg.jc_params.extra_selection,
                remake_weights=not self.cfg.jc_params.not_remake_weights,
                load_range_and_fraction=((0, 1), 1),
                file_fraction=1,
                fetch_by_files=self.cfg.jc_params.fetch_by_files,
                fetch_step=self.cfg.jc_params.fetch_step,
                infinity_mode=self.cfg.jc_params.steps_per_epoch is not None,
                in_memory=self.cfg.jc_params.in_memory,
                events_per_file=self.cfg.jc_params.events_per_file,
                name=label,
            )
        self.data_train = datasets["train"]
        self.data_test = datasets["test"]
        self.data_val = datasets["val"]

        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt/60:.2f} min")

    def _init_dataloader(self):
        self.loader_kwargs = {
            "pin_memory": True,
            "persistent_workers": self.cfg.jc_params.num_workers > 0
            and self.cfg.jc_params.steps_per_epoch is not None,
        }
        num_workers = {
            label: min(self.cfg.jc_params.num_workers, self.num_files[label])
            for label in ["train", "test", "val"]
        }
        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize,
            drop_last=True,
            num_workers=num_workers["train"],
            **self.loader_kwargs,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize,
            drop_last=True,
            num_workers=num_workers["val"],
            **self.loader_kwargs,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize,
            drop_last=False,
            num_workers=num_workers["test"],
            **self.loader_kwargs,
        )

    def _get_ypred_and_label(self, batch):
        fourmomenta = batch[0]["pf_vectors"].to(self.device)
        if self.cfg.data.features == "fourmomenta":
            scalars = torch.empty(
                fourmomenta.shape[0],
                0,
                fourmomenta.shape[2],
                device=fourmomenta.device,
                dtype=fourmomenta.dtype,
            )
        else:
            scalars = batch[0]["pf_features"].to(self.device)
        label = batch[1]["_label_"].to(self.device)
        fourmomenta, scalars, ptr = dense_to_sparse_jet(fourmomenta, scalars)
        embedding = embed_tagging_data(fourmomenta, scalars, ptr, self.cfg.data)
        y_pred, tracker, lframes = self.model(embedding)
        y_pred = y_pred[:, 0]
        return y_pred, label.to(self.dtype), tracker, lframes
