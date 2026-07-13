import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch_geometric.loader import DataLoader

from experiments.base_experiment import BaseExperiment
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow
from experiments.tagging.dataset import TopTaggingDataset
from experiments.tagging.embedding import embed_tagging_data, get_num_tagging_features
from experiments.tagging.plots import plot_mixer


class TaggingExperiment(BaseExperiment):
    """
    Base class for jet tagging experiments, focusing on binary classification
    """

    def init_physics(self):
        modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        self.momentum_dtype = torch.float64 if self.cfg.data.momentum_float64 else torch.float32

        if self.world_size > 1:
            # The tagging path is verified single-GPU only. Under DDP, attribute
            # introspection on the wrapped framesnet silently fails (no `.module`
            # unwrap anywhere): jet_frames returns None (CLS token silently rides the
            # identity frame), equivectors init_standardization is skipped, the CLS
            # no_weight_decay exemption vanishes, checkpoints get un-loadable
            # `module.*` keys, and JetClass has no rank sharding (each rank trains the
            # FULL budget). Erroring here is deliberate -- these failures are silent
            # model degradations, not crashes. Use a single GPU until the DDP rework.
            raise RuntimeError(
                "Multi-GPU tagging is disabled: the DDP wrapping breaks learned-frames "
                "introspection and checkpoint compatibility (see the audit notes in "
                "docs/diffs.md). Run with gpus=1."
            )

        # guardrail: the per-model hybrid recipes ship batchsize/lr as '???', which hydra
        # cannot enforce (an OmegaConf MISSING never overrides the values inherited from
        # tag_default), so an unfilled recipe silently composes to the 512 / 1e-3 fallback.
        if self.cfg.train and modelname.endswith(("GraphTrans", "GraphGPS")):
            # each knob checked INDEPENDENTLY: filling exactly one of the two `???`
            # placeholders must not silence the warning for the other
            unswept = []
            if float(self.cfg.training.lr) == 1e-3:
                unswept.append("lr=1e-3")
            if int(self.cfg.training.batchsize) == 512:
                unswept.append("batchsize=512")
            if unswept:
                LOGGER.warning(
                    f"{modelname} is training at the UNSWEPT family fallback "
                    f"({', '.join(unswept)}) -- did you forget to fill its recipe "
                    "from find_lr.py?"
                )

        self.cfg.model.out_channels = self.num_outputs
        if modelname in [
            "LGATr",
            "LGATrSlim",
            "LorentzNet",
            "PELICAN",
            "PELICANOfficial",
            "CGENN",
            "CGENNLGATrGraphTrans",
            "LorentzNetLGATrSlimGraphTrans",
            "CGENNLGATrGraphGPS",
            "LorentzNetLGATrSlimGraphGPS",
        ]:
            # Lorentz-equivariance by internal representations
            in_s_channels = self.extra_scalars
            in_s_channels += get_num_tagging_features(
                tagging_features=self.cfg.data.tagging_features
            )
            if modelname in ["LGATr", "LGATrSlim"]:
                self.cfg.model.net.in_s_channels = 0 if self.cfg.model.mean_aggregation else 1
                self.cfg.model.net.in_s_channels += in_s_channels
            elif modelname == "LorentzNet":
                self.cfg.model.net.n_scalar = in_s_channels
            elif modelname == "PELICAN":
                self.cfg.model.net.in_channels_rank1 = in_s_channels
            elif modelname == "PELICANOfficial":
                self.cfg.model.net.num_scalars = in_s_channels
            elif modelname == "CGENN":
                # CGENN cant handle zero scalar inputs -> give 1 input with zeros
                self.cfg.model.net.in_features_h = 1 + in_s_channels
            elif modelname == "CGENNLGATrGraphTrans":
                # zero scalar inputs doesn't really happen so I ignore it out of practicality
                self.cfg.model.net.in_s_channels = in_s_channels
            elif modelname == "LorentzNetLGATrSlimGraphTrans":
                self.cfg.model.net.in_s_channels = in_s_channels
            elif modelname == "CGENNLGATrGraphGPS":
                # same thing
                self.cfg.model.net.in_s_channels = in_s_channels
            elif modelname == "LorentzNetLGATrSlimGraphGPS":
                self.cfg.model.net.in_s_channels = in_s_channels

            # doesn't affect results and never needed
            self.cfg.data.boost_jet = False
        elif modelname in [
            "Transformer",
            "ParticleTransformer",
            "GraphNet",
            "ParticleNet",
            "MIParticleTransformer",
            "ParticleNetParTGraphTrans",
            "ParticleNetParTGraphGPS",
            "PlainGraphTrans",
            "PlainGraphGPS",
        ]:
            # Non-equivariant or canonicalization
            self.cfg.model.in_channels = 7 + self.extra_scalars
            if self.cfg.model.add_fourmomenta_backbone:
                self.cfg.model.in_channels += 4

            if modelname == "Transformer":
                self.cfg.model.in_channels += 0 if self.cfg.model.mean_aggregation else 1
            elif modelname == "GraphNet":
                self.cfg.model.net.num_edge_attr = 1 if self.cfg.model.include_edges else 0
            elif modelname == "ParticleNet":
                self.cfg.model.net.hidden_reps_list[0] = f"{self.cfg.model.in_channels}x0n"

            # decide which entries to use for the framesnet
            if "equivectors" in self.cfg.model.framesnet:
                num_tagging_features = get_num_tagging_features(
                    tagging_features=self.cfg.data.tagging_features
                )
                self.cfg.model.framesnet.equivectors.num_scalars = self.extra_scalars
                self.cfg.model.framesnet.equivectors.num_scalars += num_tagging_features
            else:
                # not allowed, because the network is not Lorentz-equivariant
                self.cfg.data.boost_jet = False
        else:
            raise NotImplementedError(f"Model {modelname} not implemented")

    def init_data(self):
        raise NotImplementedError

    def _init_data(self, Dataset, data_path):
        LOGGER.info(f"Creating {Dataset.__name__} from {data_path}")
        t0 = time.time()
        self.data_train = Dataset()
        self.data_test = Dataset()
        self.data_val = Dataset()
        kwargs = dict(
            network_float64=self.cfg.use_float64,
            momentum_float64=self.cfg.data.momentum_float64,
        )
        self.data_train.load_data(data_path, "train", **kwargs)
        self.data_test.load_data(data_path, "test", **kwargs)
        self.data_val.load_data(data_path, "val", **kwargs)
        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt / 60:.2f} min")

    def _init_dataloader(self):
        trn_sampler = torch.utils.data.DistributedSampler(
            self.data_train,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        tst_sampler = torch.utils.data.DistributedSampler(
            self.data_test,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )
        val_sampler = torch.utils.data.DistributedSampler(
            self.data_val,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )

        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize // self.world_size,
            sampler=trn_sampler,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=tst_sampler,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=val_sampler,
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

        self.init_standardization()

    def init_standardization(self):
        if hasattr(self.model, "init_standardization"):
            batch = next(iter(self.train_loader))
            fourmomenta, scalars, ptr, _ = self._extract_batch(batch)
            embedding = embed_tagging_data(
                fourmomenta,
                scalars,
                ptr,
                self.cfg.data,
            )
            self.model.init_standardization(embedding["fourmomenta"], embedding["ptr"])

    def _init_optimizer(self, param_groups=None):
        # ParT/weaver weight-decay grouping. Its ONLY effect beyond the base default
        # (base_experiment._init_optimizer) is excluding the learnable CLS token(s) from
        # weight decay via net.no_weight_decay() -- the base already exempts norms/biases
        # (ndim<=1) and groups the framesnet (even splitting its biases off). So only models
        # with a class token (a non-empty no_weight_decay()) belong here. The mean-pool
        # GraphGPS models have none -- CGENN/LorentzNet GPS return set() and the plain/PNet
        # GPS define no such method -- so they correctly fall through to the base default.
        # Caller-supplied groups (the finetune experiment's carefully split
        # backbone/head lrs) must NOT be clobbered by the name match below.
        if param_groups is None and self.cfg.model.net._target_.rsplit(".", 1)[-1] in [
            "ParticleTransformer",
            "MIParticleTransformer",
            "ParticleNetParTGraphTrans",
            "LorentzNetLGATrSlimGraphTrans",
            "CGENNLGATrGraphTrans",
            "PlainGraphTrans",
        ]:
            # special treatment for ParT, see
            # https://github.com/hqucms/weaver-core/blob/dev/custom_train_eval/weaver/train.py#L464
            decay, no_decay = {}, {}
            for name, param in self.model.net.named_parameters():
                if not param.requires_grad:
                    continue
                if (
                    len(param.shape) == 1
                    or name.endswith(".bias")
                    or (
                        hasattr(self.model.net, "no_weight_decay")
                        and name in self.model.net.no_weight_decay()
                    )
                ):
                    no_decay[name] = param
                else:
                    decay[name] = param
            decay_1x, no_decay_1x = list(decay.values()), list(no_decay.values())
            param_groups = [
                {
                    "params": no_decay_1x,
                    "weight_decay": 0.0,
                    "lr": self.cfg.training.lr,
                },
                {
                    "params": decay_1x,
                    "weight_decay": self.cfg.training.weight_decay,
                    "lr": self.cfg.training.lr,
                },
                {
                    "params": self.model.framesnet.parameters(),
                    "weight_decay": self.cfg.training.weight_decay_framesnet,
                    "lr": self.cfg.training.lr * self.cfg.training.lr_factor_framesnet,
                },
            ]

        super()._init_optimizer(param_groups=param_groups)

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
                        loader_dict[set_label], f"{set_label}_ema", mode="eval"
                    )

                self._evaluate_single(loader_dict[set_label], set_label, mode="eval")

            else:
                self.results[set_label] = self._evaluate_single(
                    loader_dict[set_label], set_label, mode="eval"
                )

    @torch.no_grad()
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
        for batch in loader:
            y_pred, label, _, _ = self._get_ypred_and_label(batch)
            labels_true.append(label.cpu().float())
            labels_predict.append(y_pred.cpu().float())
        labels_true, labels_predict = torch.cat(labels_true), torch.cat(labels_predict)

        # bce loss (from the raw logits)
        metrics["loss"] = torch.nn.functional.binary_cross_entropy_with_logits(
            labels_predict, labels_true
        ).item()
        labels_predict = torch.nn.functional.sigmoid(labels_predict)
        if mode == "eval":
            # store the sigmoid PROBABILITIES, not the logits: plot_score histograms
            # metrics["labels_predict"] over [0, 1] (matplotlib silently drops
            # out-of-range values, so storing logits made score.pdf meaningless)
            metrics["labels_true"], metrics["labels_predict"] = (
                labels_true,
                labels_predict,
            )
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
            # per-trial scalars; accumulated across run_idx so the table can show
            # mean +- std error bars when an experiment has several trials/seeds
            row = {
                # float() casts: numpy scalars are not JSON-serializable
                "accuracy": float(metrics["accuracy"]),
                "auc": float(metrics["auc"]),
                "rej03": float(metrics["rej03"]),
                "rej05": float(metrics["rej05"]),
                "rej08": float(metrics["rej08"]),
                "train_time": getattr(self, "train_time", None),
            }
            self._log_table_row(
                loader,
                title,
                row,
                [
                    ("accuracy", ".4f"),
                    ("auc", ".4f"),
                    ("rej03", ".0f"),
                    ("rej05", ".0f"),
                    ("rej08", ".0f"),
                ],
            )
        return metrics

    def _frames_description(self):
        """Frames-column cell: the framesnet class name, disambiguated where one class
        serves several config variants (all four random-frames configs -- randomlorentz /
        randomrotation / randomxyrotation / randomztransform -- instantiate the same
        ``RandomFrames`` class; without the transform_type they would collide into a
        single aggregate_table row)."""
        fn = self.model.framesnet
        name = type(fn).__name__
        kind = getattr(fn, "transform_type", None)
        return f"{name}({kind})" if kind is not None else name

    def _log_table_row(self, loader, title, row, metric_fmts):
        """Emit the parseable ``table <title>:`` results row for this split.

        ``row`` holds this trial's scalar metrics (every key in ``metric_fmts`` plus
        ``train_time``); ``metric_fmts`` is the ordered ``(key, fmt)`` list of metric
        columns, so subclasses with different metric sets (e.g. JetClass's per-class
        rejections) share the surrounding model/frames/params/flops/kNN plumbing and
        the multi-trial mean +- std accumulation (see ``_collect_table_rows``).
        """
        modelname = type(self.model.net).__name__
        framesString = self._frames_description()
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        knn = self._knn_description()
        flops = self._count_flops(loader)
        flops_str = f"{flops:.3e}" if flops is not None else "n/a"

        rows = self._collect_table_rows(title, row)
        n_trials = len(rows)

        def cell(key, fmt):
            vals = [r[key] for r in rows if r.get(key) is not None]
            if not vals:
                return "n/a"
            if len(vals) == 1:
                return format(vals[0], fmt)
            arr = np.asarray(vals, dtype=float)
            return f"${format(arr.mean(), fmt)} \\pm {format(arr.std(ddof=1), fmt)}$"

        trials = f" [{n_trials} trials]" if n_trials > 1 else ""
        time_cell = cell("train_time", ".0f")
        if time_cell != "n/a":
            time_cell += "s"
        metric_cells = " & ".join(cell(key, fmt) for key, fmt in metric_fmts)
        # columns: model & frames & iters[trials] & params & <metrics...> & traintime
        #          & flops & knn
        LOGGER.info(
            f"table {title}: {modelname} & {framesString}"
            f" & {self.cfg.training.iterations}{trials}"
            f" & {num_parameters} & {metric_cells}"
            f" & {time_cell} & {flops_str} & {knn} \\\\"
        )

    def _collect_table_rows(self, title, row):
        """Persist this run's table metrics and return all trials in the run dir.

        Multiple trials/seeds are launched as successive run_idx that share one
        run directory (fresh-trial warm starts: warm_start_load=false, so each
        trial trains from its own random init -- see GUIDE.md section 8), each a
        separate process, launched SEQUENTIALLY (the JSON accumulation below is not
        locked against concurrent writers). We accumulate their scalar metrics in a
        JSON file so the final table reports mean +- std automatically. With
        save=False (e.g. tests) nothing is written and only the current run is
        returned.

        Rows are keyed by trial LINEAGE: a continue-training warm start
        (warm_start_load=true) extends an existing trial rather than adding one, so
        its row REPLACES the parent trial's row instead of appending a duplicate
        (which would fake an extra trial and shrink the std).
        """
        if not self.cfg.save:
            return [row]
        path = os.path.join(self.cfg.run_dir, f"table_metrics_{title}.json")
        rows = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    rows = json.load(f)
            except (json.JSONDecodeError, OSError):
                rows = []
        if not self.cfg.train:
            # eval-only rerun (e.g. a warm-start reload): it re-evaluates an ALREADY-COUNTED
            # trial, so appending would duplicate that trial's row and shrink the std. Just
            # report the accumulated trials (or this run alone if none are persisted yet).
            return rows if rows else [row]
        run_idx = int(self.cfg.run_idx or 0)
        lineage = run_idx
        if self.warm_start and self.warm_load:
            # continue-training: inherit the parent trial's lineage root (chained
            # continuations then still collapse onto one row)
            parent = int(self.cfg.warm_start_idx)
            parent_row = next((r for r in rows if r.get("run_idx") == parent), None)
            lineage = parent_row.get("lineage", parent) if parent_row is not None else parent
        row = dict(row, run_idx=run_idx, lineage=lineage)
        # legacy rows without lineage bookkeeping are kept as separate trials
        rows = [r for r in rows if r.get("lineage", object()) != lineage]
        rows.append(row)
        try:
            with open(path, "w") as f:
                json.dump(rows, f)
        except OSError as e:
            LOGGER.warning(f"Could not persist table metrics to {path}: {e}")
        return rows

    def _knn_description(self):
        """Short label for the model's kNN graph metric, or '-' if it has none."""
        net = getattr(self.model, "net", None)
        metric = getattr(net, "knn_metric", None)
        if metric is not None:
            return metric
        if type(net).__name__ == "ParticleNet":
            return "deltaR"  # L2 on (eta, phi)
        return "-"

    @torch.no_grad()
    def _count_flops(self, loader):
        """Forward FLOPs for a single jet (batchsize 1), as in test_tag_flops."""
        try:
            from torch.utils.flop_counter import FlopCounterMode

            batch = next(iter(loader)).clone().to(self.device)
            n = int(batch.ptr[1].item())  # keep only the first jet
            batch.x = batch.x[:n]
            batch.scalars = batch.scalars[:n]
            batch.batch = batch.batch[:n]
            batch.ptr = batch.ptr[:2]
            batch.label = batch.label[:1]
            with FlopCounterMode(display=False) as flop_counter:
                self._get_ypred_and_label(batch)
            return flop_counter.get_total_flops()
        except Exception as e:
            LOGGER.warning(f"FLOPs counting failed: {e}")
            return None

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        title = type(self.model.net).__name__
        LOGGER.info(f"Creating plots in {plot_path}")

        if (
            self.cfg.evaluation.save_roc
            and self.cfg.evaluate
            and ("test" in self.cfg.evaluation.eval_set)
        ):
            file = f"{plot_path}/roc.txt"
            roc = np.stack((self.results["test"]["fpr"], self.results["test"]["tpr"]), axis=-1)
            np.savetxt(file, roc)

        plot_dict = {}
        if self.cfg.evaluate and ("test" in self.cfg.evaluation.eval_set):
            plot_dict = {"results_test": self.results["test"]}
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
            plot_dict["grad_norm"] = torch.stack(self.grad_norm_train).cpu()
            plot_dict["grad_norm_frames"] = torch.stack(self.grad_norm_frames).cpu()
            plot_dict["grad_norm_net"] = torch.stack(self.grad_norm_net).cpu()
            for key, value in self.train_metrics.items():
                plot_dict[key] = value
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    def _init_loss(self):
        self.loss = torch.nn.BCEWithLogitsLoss()

    # overwrite _validate method to compute metrics over the full validation set
    def _validate(self, step):
        if self.ema is not None:
            with self.ema.average_parameters():
                metrics = self._evaluate_single(self.val_loader, "val", mode="val", step=step)
        else:
            metrics = self._evaluate_single(self.val_loader, "val", mode="val", step=step)
        self.val_loss.append(metrics["loss"])
        # per-validation (step, loss, accuracy) history for the end-of-training
        # loss-vs-accuracy selection cross-check (_log_checkpoint_selection)
        if not hasattr(self, "val_selection_history"):
            self.val_selection_history = []
        self.val_selection_history.append((step, metrics["loss"], metrics["accuracy"]))
        # Best-checkpoint selection metric (es_load_best_model). Default 'loss' (lower=better).
        # 'accuracy' keeps the highest-val-accuracy checkpoint instead; we return 1 - accuracy so
        # the train loop's "lower is better" comparison is unchanged. Loss is still logged above,
        # so only which checkpoint is kept/reported changes.
        if self.cfg.training.get("best_model_metric", "loss") == "accuracy":
            return 1.0 - metrics["accuracy"]
        return metrics["loss"]

    def _log_checkpoint_selection(self, best_step):
        """Cross-check the kept checkpoint against selection-by-accuracy.

        Runs after the best-val-loss restore. Selection-by-loss and -by-accuracy usually
        agree; small argmax divergence is common late in training (accuracy is a thresholded
        count, so near-ties are settled by counting noise), but a MATERIAL accuracy gap is
        rare and worth eyes -- it is the one signal that selection-by-loss cost accuracy.
        Only meaningful under the default best_model_metric='loss' (under 'accuracy' the
        kept checkpoint IS the accuracy argmax).
        """
        hist = getattr(self, "val_selection_history", [])
        if len(hist) < 2 or self.cfg.training.get("best_model_metric", "loss") != "loss":
            return
        by_loss = min(hist, key=lambda t: t[1])
        by_acc = max(hist, key=lambda t: t[2])
        if by_acc[0] == by_loss[0]:
            LOGGER.info(
                f"Checkpoint selection: best val loss and best val accuracy agree "
                f"(it {by_loss[0]}: loss {by_loss[1]:.5f}, acc {by_loss[2]:.5f})."
            )
            return
        gap = by_acc[2] - by_loss[2]
        msg = (
            f"Checkpoint selection diverges: kept best-val-LOSS it {by_loss[0]} "
            f"(loss {by_loss[1]:.5f}, acc {by_loss[2]:.5f}); best-val-ACCURACY was "
            f"it {by_acc[0]} (loss {by_acc[1]:.5f}, acc {by_acc[2]:.5f}), "
            f"accuracy gap {gap:+.5f}."
        )
        # ~1 sigma of accuracy counting noise on a few-1e5-jet val set; below this the
        # divergence is expected tie-breaking, above it flag for eyes
        if gap > 5e-4:
            LOGGER.warning(
                msg + " Material gap -- consider a best_model_metric=accuracy cross-check run."
            )
        else:
            LOGGER.info(msg + " Within counting noise; selection-by-loss stands.")

    def _batch_loss(self, batch):
        y_pred, label, tracker, _ = self._get_ypred_and_label(batch)
        loss = self.loss(y_pred, label)

        metrics = tracker
        return loss, metrics

    def _extract_batch(self, batch):
        batch = batch.to(self.device)
        fourmomenta = batch.x.to(self.momentum_dtype)
        scalars = batch.scalars.to(self.dtype)
        ptr = batch.ptr
        label = batch.label.to(self.dtype)
        return fourmomenta, scalars, ptr, label

    def _get_ypred_and_label(self, batch):
        fourmomenta, scalars, ptr, label = self._extract_batch(batch)
        embedding = embed_tagging_data(
            fourmomenta,
            scalars,
            ptr,
            self.cfg.data,
        )
        embedding["num_graphs"] = label.shape[0]
        y_pred, tracker, frames = self.model(embedding)
        if isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            y_pred = y_pred[:, 0]
        return y_pred, label, tracker, frames

    def _init_metrics(self):
        return {
            "reg_collinear": [],
            "reg_coplanar": [],
            "reg_lightlike": [],
            "reg_gammamax": [],
            "gamma_mean": [],
            "gamma_max": [],
        }


class TopTaggingExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.extra_scalars = 0

    def init_data(self):
        data_path = os.path.join(self.cfg.data.data_dir, f"toptagging_{self.cfg.data.dataset}.npz")
        self._init_data(TopTaggingDataset, data_path)
