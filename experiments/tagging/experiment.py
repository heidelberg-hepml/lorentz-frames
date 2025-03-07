import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import os, time
from omegaconf import open_dict

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

from experiments.base_experiment import BaseExperiment
from experiments.tagging.dataset import TopTaggingDataset
from experiments.tagging.embedding import embed_tagging_data
from experiments.tagging.plots import plot_mixer
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow


class TaggingExperiment(BaseExperiment):
    """
    Base class for jet tagging experiments, focusing on binary classification
    """

    def init_physics(self):
        with open_dict(self.cfg):

            # decide which entries to use for the net
            in_channels = 7
            self.cfg.model.in_channels = in_channels
            LOGGER.info(
                f"Net: Input: {in_channels}; Output: {self.cfg.model.out_channels} "
            )

            # decide which entries to use for the lframesnet
            in_nodes = 0

            if "in_nodes" in self.cfg.model.lframesnet:
                if self.cfg.model.add_tagging_features_lframesnet:
                    in_nodes += 7
                self.cfg.model.lframesnet.in_nodes = in_nodes

            if self.cfg.model.net._target_.rsplit(".", 1)[-1] == "TFGraphNet":
                self.cfg.model.net.num_edge_attr = (
                    1 if self.cfg.model.include_edges else 0
                )

            if (
                self.cfg.model._target_.rsplit(".", 1)[-1]
                == "BaselineParticleNetWrapper"
            ):
                # Note: cfg.data.add_scalar_features not supported for net inputs; in_channels currently hard-coded
                if (
                    self.cfg.data.beam_reference is not None
                    or self.cfg.data.add_time_reference
                ):
                    LOGGER.warning(
                        "Spurions not supported for BaselineParticleNetWrapper (yield nan/inf in get_tagging_features), removing them"
                    )
                    self.cfg.data.beam_reference = None
                    self.cfg.data.add_time_reference = False

    def init_data(self):
        raise NotImplementedError

    def _init_data(self, Dataset, data_path):
        LOGGER.info(f"Creating {Dataset.__name__} from {data_path}")
        t0 = time.time()
        kwargs = {"rescale_data": self.cfg.data.rescale_data}
        self.data_train = Dataset(**kwargs)
        self.data_test = Dataset(**kwargs)
        self.data_val = Dataset(**kwargs)
        self.data_train.load_data(data_path, "train")
        self.data_test.load_data(data_path, "test")
        self.data_val.load_data(data_path, "val")
        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt/60:.2f} min")

        # only take 10000 batches for performance reasons
        if self.cfg.data.standardize:
            self.model.init_standardization(
                batch=Batch.from_data_list(
                    self.data_train.data_list[
                        : min(10000, len(self.data_train.data_list))
                    ]
                ).to(self.device)
            )

    def _init_dataloader(self):
        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

    def evaluate(self):
        self.results = {}
        loader_dict = {
            "train": self.train_loader,
            "test": self.test_loader,
            "val": self.val_loader,
        }
        for set_label in self.cfg.evaluation.eval_set:
            if self.ema is not None:
                with self.ema.average_parameters():
                    self.results[set_label] = self._evaluate_single(
                        loader_dict[set_label], set_label, mode="eval"
                    )

                self._evaluate_single(
                    loader_dict[set_label], f"{set_label}_noema", mode="eval"
                )

            else:
                self.results[set_label] = self._evaluate_single(
                    loader_dict[set_label], set_label, mode="eval"
                )

    def _evaluate_single(self, loader, title, mode, step=None):
        assert mode in ["val", "eval"]

        if mode == "eval":
            LOGGER.info(
                f"### Starting to evaluate model on {title} dataset with "
                f"{len(loader.dataset)} elements, batchsize {loader.batch_size} ###"
            )
        metrics = {}

        # predictions
        labels_true, labels_predict = [], []
        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()
        with torch.no_grad():
            for batch in loader:
                y_pred, label, _ = self._get_ypred_and_label(batch)
                y_pred = torch.nn.functional.sigmoid(y_pred)
                labels_true.append(label.cpu().float())
                labels_predict.append(y_pred.cpu().float())
        labels_true, labels_predict = torch.cat(labels_true), torch.cat(labels_predict)
        if mode == "eval":
            metrics["labels_true"], metrics["labels_predict"] = (
                labels_true,
                labels_predict,
            )

        # bce loss
        metrics["loss"] = torch.nn.functional.binary_cross_entropy(
            labels_predict, labels_true
        ).item()
        labels_true, labels_predict = labels_true.numpy(), labels_predict.numpy()

        # accuracy
        metrics["accuracy"] = accuracy_score(labels_true, np.round(labels_predict))
        if mode == "eval":
            LOGGER.info(f"Accuracy on {title} dataset: {metrics['accuracy']:.4f}")

        # roc (fpr = epsB, tpr = epsS)
        fpr, tpr, th = roc_curve(labels_true, labels_predict)
        if mode == "eval":
            metrics["fpr"], metrics["tpr"] = fpr, tpr
        metrics["auc"] = roc_auc_score(labels_true, labels_predict)
        if mode == "eval":
            LOGGER.info(f"AUC score on {title} dataset: {metrics['auc']:.4f}")

        # 1/epsB at fixed epsS
        def get_rej(epsS):
            idx = np.argmin(np.abs(tpr - epsS))
            return 1 / fpr[idx]

        metrics["rej03"] = get_rej(0.3)
        metrics["rej05"] = get_rej(0.5)
        metrics["rej08"] = get_rej(0.8)
        if mode == "eval":
            LOGGER.info(
                f"Rejection rate {title} dataset: {metrics['rej03']:.0f} (epsS=0.3), "
                f"{metrics['rej05']:.0f} (epsS=0.5), {metrics['rej08']:.0f} (epsS=0.8)"
            )

        if self.cfg.use_mlflow:
            for key, value in metrics.items():
                if key in ["labels_true", "labels_predict", "fpr", "tpr"]:
                    # do not log matrices
                    continue
                name = f"{mode}.{title}" if mode == "eval" else "val"
                log_mlflow(f"{name}.{key}", value, step=step)

        if mode == "eval":
            lframeString = type(self.model.lframesnet).__name__
            num_parameters = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            LOGGER.info(
                f"table {title}: {lframeString} ({self.cfg.training.iterations} epochs)"
                f" & {num_parameters} & {metrics['accuracy']:.4f}&{metrics['auc']:.4f}"
                f" & {metrics['rej03']:.0f}&{metrics['rej05']:.0f}&{metrics['rej08']:.0f} \\\\"
            )
        return metrics

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path)
        title = type(self.model.net).__name__
        LOGGER.info(f"Creating plots in {plot_path}")

        if (
            self.cfg.evaluation.save_roc
            and self.cfg.evaluate
            and ("test" in self.cfg.evaluation.eval_set)
        ):
            file = f"{plot_path}/roc.txt"
            roc = np.stack(
                (self.results["test"]["fpr"], self.results["test"]["tpr"]), axis=-1
            )
            np.savetxt(file, roc)

        plot_dict = {}
        if self.cfg.evaluate and ("test" in self.cfg.evaluation.eval_set):
            plot_dict = {"results_test": self.results["test"]}
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
            plot_dict["train_metrics"] = self.train_metrics
            plot_dict["val_metrics"] = self.val_metrics
            plot_dict["grad_norm"] = self.grad_norm_train
            plot_dict["grad_norm_lframes"] = self.grad_norm_lframes
            plot_dict["grad_norm_net"] = self.grad_norm_net
            for key, value in self.train_metrics.items():
                plot_dict[key] = value
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    def _init_loss(self):
        self.loss = torch.nn.BCEWithLogitsLoss()

    # overwrite _validate method to compute metrics over the full validation set
    def _validate(self, step):
        if self.ema is not None:
            with self.ema.average_parameters():
                metrics = self._evaluate_single(
                    self.val_loader, "val", mode="val", step=step
                )
        else:
            metrics = self._evaluate_single(
                self.val_loader, "val", mode="val", step=step
            )
        self.val_loss.append(metrics["loss"])
        return metrics["loss"]

    def _batch_loss(self, batch):
        y_pred, label, tracker = self._get_ypred_and_label(batch)
        loss = self.loss(y_pred, label)
        assert torch.isfinite(loss).all()

        metrics = tracker
        return loss, metrics

    def _get_ypred_and_label(self, batch):
        batch = batch.to(self.device)
        embedding = embed_tagging_data(
            batch.x,
            batch.scalars,
            batch.ptr,
            self.cfg.data,
        )
        y_pred, tracker = self.model(embedding)
        y_pred = y_pred[:, 0]
        return y_pred, batch.label.to(self.dtype), tracker

    def _init_metrics(self):
        return {"reg_collinear": [], "reg_coplanar": [], "reg_lightlike": []}


class TopTaggingExperiment(TaggingExperiment):
    def __init__(self, cfg):
        super().__init__(cfg)
        with open_dict(self.cfg):
            self.cfg.model.out_channels = 1

            # move argument into model config
            self.cfg.model.add_tagging_features_lframesnet = (
                self.cfg.data.add_tagging_features_lframesnet
            )

    def init_data(self):
        data_path = os.path.join(
            self.cfg.data.data_dir, f"toptagging_{self.cfg.data.dataset}.npz"
        )
        self._init_data(TopTaggingDataset, data_path)
