import numpy as np
import torch
from torch_geometric.loader import DataLoader

import os, time
from omegaconf import open_dict

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

from experiments.base_experiment import BaseExperiment
from experiments.toptagging.dataset import TopTaggingDataset
from experiments.toptagging.embedding import embed_tagging_data
from experiments.toptagging.plots import plot_mixer
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

MODEL_TITLE_DICT = {
    "ProtoNet": "ProtoNet",
    "NonEquiNet": "NoneEquiNet",
}

UNITS = 20  # We use units of 20 GeV for all tagging experiments


class TaggingExperiment(BaseExperiment):
    """
    Generalization of all tagging experiments
    """

    def init_physics(self):
        # dynamically extend dict
        with open_dict(self.cfg):
            protonet_name = "experiments.toptagging.wrappers.ProtoNetWrapper"
            nonequinet_name = "experiments.toptagging.wrappers.NonEquiNetWrapper"
            assert self.cfg.model._target_ in [
                protonet_name,
                nonequinet_name,
            ]

            # global token?
            if self.cfg.model._target_ in [
                protonet_name,
                nonequinet_name,
            ]:
                self.cfg.data.include_global_token = not self.cfg.model.mean_aggregation

    def init_data(self):
        raise NotImplementedError

    def _init_data(self, Dataset, data_path):
        LOGGER.info(f"Creating {Dataset.__name__} from {data_path}")
        t0 = time.time()
        kwargs = {"rescale_data": self.cfg.data.rescale_data}
        self.data_train = Dataset(**kwargs)
        self.data_test = Dataset(**kwargs)
        self.data_val = Dataset(**kwargs)
        self.data_train.load_data(data_path, "train", data_scale=UNITS)
        self.data_test.load_data(data_path, "test", data_scale=UNITS)
        self.data_val.load_data(data_path, "val", data_scale=UNITS)
        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt/60:.2f} min")

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

        # this is a bit ugly, but it does the job
        if self.ema is not None:
            with self.ema.average_parameters():
                self.results["train"] = self._evaluate_single(
                    self.train_loader, "train", mode="eval"
                )
                self.results["val"] = self._evaluate_single(
                    self.val_loader, "val", mode="eval"
                )
                self.results["test"] = self._evaluate_single(
                    self.test_loader, "test", mode="eval"
                )

            self._evaluate_single(self.train_loader, "train_noema", mode="eval")
            self._evaluate_single(self.val_loader, "val_noema", mode="eval")
            self._evaluate_single(self.test_loader, "test_noema", mode="eval")

        else:
            self.results["train"] = self._evaluate_single(
                self.train_loader, "train", mode="eval"
            )
            self.results["val"] = self._evaluate_single(
                self.val_loader, "val", mode="eval"
            )
            self.results["test"] = self._evaluate_single(
                self.test_loader, "test", mode="eval"
            )

    def _evaluate_single(self, loader, title, mode, step=None):
        assert mode in ["val", "eval"]
        # re-initialize dataloader to make sure it is using the evaluation batchsize (makes a difference for trainloader)
        loader = DataLoader(
            dataset=loader.dataset,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        if mode == "eval":
            LOGGER.info(
                f"### Starting to evaluate model on {title} dataset with "
                f"{len(loader.dataset.data_list)} elements, batchsize {loader.batch_size} ###"
            )
        metrics = {}

        # predictions
        labels_true, labels_predict = [], []
        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()
        with torch.no_grad():
            for batch in loader:
                y_pred, label = self._get_ypred_and_label(batch)
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
        metrics["bce"] = torch.nn.functional.binary_cross_entropy(
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
            aggregator = (
                "mean aggregation"
                if self.cfg.model.mean_aggregation == True
                else "global token"
            )
            match self.cfg.model.lframesnet.approach:
                case "learned_gramschmidt":
                    lframeString = "Gram-Schmidt"
                case "identity":
                    lframeString = "Identity"
                case "random_global":
                    lframeString = "Random Global"
                case "random_local":
                    lframeString = "Random Local"
                case "3nn":
                    lframeString = "3nn"
                case _:
                    lframeString = self.cfg.model.lframesnet.approach
            num_parameters = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            if (
                self.cfg.model.radial_module._target_
                == "tensorframes.nn.embedding.radial.TrivialRadialEmbedding"
            ):
                learnableString = "no embedding"
            elif self.cfg.model.radial_module.is_learnable == True:
                learnableString = "learned"
            elif self.cfg.model.radial_module.is_learnable == False:
                learnableString = "\mathbb{1}"
            else:
                learnableString = "other"

            LOGGER.info(
                f"table {title}: {lframeString} with {aggregator} ({self.cfg.training.iterations} epochs)&{num_parameters}&{metrics['accuracy']:.4f}&{metrics['auc']:.4f}&{metrics['rej03']:.0f}&{metrics['rej05']:.0f}&{metrics['rej08']:.0f}&{learnableString}\\"
            )
        return metrics

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path)
        title = MODEL_TITLE_DICT[type(self.model.net).__name__]
        LOGGER.info(f"Creating plots in {plot_path}")

        if self.cfg.evaluate and self.cfg.evaluation.save_roc:
            file = f"{plot_path}/roc.txt"
            roc = np.stack(
                (self.results["test"]["fpr"], self.results["test"]["tpr"]), axis=-1
            )
            np.savetxt(file, roc)

        plot_dict = {}
        if self.cfg.evaluate:
            plot_dict = {"results_test": self.results["test"]}
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
            plot_dict["train_metrics"] = self.train_metrics
            plot_dict["val_metrics"] = self.val_metrics
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
        self.val_loss.append(metrics["bce"])
        return metrics["bce"]

    def _batch_loss(self, batch):
        y_pred, label = self._get_ypred_and_label(batch)
        loss = self.loss(y_pred, label)
        assert torch.isfinite(loss).all()

        metrics = {}
        return loss, metrics

    def _get_ypred_and_label(self, batch):
        batch = batch.to(self.device)
        embedding = embed_tagging_data(batch.x, batch.scalars, batch.ptr, self.cfg.data)
        y_pred = self.model(embedding)
        return y_pred, batch.label.to(self.dtype)

    def _init_metrics(self):
        return {}


class TopTaggingExperiment(TaggingExperiment):
    def __init__(self, cfg):
        super().__init__(cfg)

        fourvector_reps = "1x0n+1x1n"  # this is a representation of a fourvector with the current tensorframes
        self.in_reps = fourvector_reps  # energy-momentum vector

        if self.cfg.data.add_scalar_features:
            self.in_reps = "7x0n+" + self.in_reps  # other scalar features
            if self.cfg.model.mean_aggregation is False:
                    self.in_reps = str(self.cfg.data.num_global_tokens)+"x0n+" + self.in_reps

        if not self.cfg.data.beam_token:
            if self.cfg.data.beam_reference in ["lightlike", "spacelike", "timelike"]:
                self.in_reps = self.in_reps + "+" + fourvector_reps  # spurions

                if self.cfg.data.two_beams:
                    self.in_reps = (
                        self.in_reps + "+" + fourvector_reps
                    )  # two beam spurions

            if self.cfg.data.add_time_reference:
                self.in_reps = self.in_reps + "+" + fourvector_reps  # time spurion

        LOGGER.info(f"Using the input representation: {self.in_reps}")
        self.cfg.model.in_reps = self.in_reps

    def init_data(self):
        data_path = os.path.join(
            self.cfg.data.data_dir, f"toptagging_{self.cfg.data.dataset}.npz"
        )
        self._init_data(TopTaggingDataset, data_path)
