import os, time
import numpy as np
import torch

from experiments.base_experiment import BaseExperiment
from experiments.amplitudes.utils import (
    preprocess_amplitude,
    undo_preprocess_amplitude,
)
from experiments.amplitudes.constants import PARTICLE_TYPE, DATASET_TITLE, get_mass
from experiments.amplitudes.plots import plot_mixer
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

from tensorframes.utils.transforms import (
    rand_lorentz,
    rand_rotation,
    rand_xyrotation,
    rand_boost,
)

MODEL_TITLE = {
    "TFTransformer": "Tr",
    "MLP": "MLP",
    "TFGraphNet": "GN",
    "GATr": "GATr",
    "DSI": "DSI",
}


class AmplitudeExperiment(BaseExperiment):
    def init_physics(self):
        self.dataset = self.cfg.data.dataset.rsplit("_")[0]
        particle_type = PARTICLE_TYPE[self.dataset]
        if not self.cfg.data.permutation_symmetry:
            particle_type = list(range(len(particle_type)))
        num_particle_types = max(particle_type) + 1

        modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        learnable_lframesnet = "equivectors" in self.cfg.model.lframesnet
        self.cfg.model.particle_type = particle_type

        if modelname == "TFTransformer":
            self.cfg.model.net.in_channels = num_particle_types + 4
        elif modelname == "TFGraphNet":
            assert self.cfg.model.include_nodes or self.cfg.model.include_edges
            self.cfg.model.net.num_edge_attr = 1 if self.cfg.model.include_edges else 0
            self.cfg.model.net.in_channels = num_particle_types
            if self.cfg.model.include_nodes:
                self.cfg.model.net.in_channels += 4
        elif modelname == "MLP":
            self.cfg.model.net.in_shape = 4 * len(particle_type)
        elif modelname == "GATr":
            assert not learnable_lframesnet, "GATr is no tensorframes model"
            self.cfg.model.net.in_s_channels = num_particle_types
        elif modelname == "DSI":
            assert not learnable_lframesnet, "DSI is no tensorframes model"
            self.cfg.model.net.type_token_list = particle_type
        else:
            raise ValueError(f"Model {modelname} not implemented")

        if learnable_lframesnet:
            self.cfg.model.lframesnet.equivectors.num_scalars = num_particle_types
        LOGGER.info(f"Using particle_type={particle_type}")

    def init_data(self):
        LOGGER.info(f"Using dataset={self.cfg.data.dataset}")

        # load data
        data_path = os.path.join(
            self.cfg.data.data_path, f"{self.cfg.data.dataset}.npy"
        )
        assert os.path.exists(data_path), f"data_path {data_path} does not exist"
        data_raw = np.load(data_path)
        data_raw = torch.tensor(data_raw, dtype=self.dtype)
        LOGGER.info(f"Loaded data with shape {data_raw.shape} from {data_path}")

        # bring data into correct shape
        if self.cfg.data.subsample is not None:
            assert self.cfg.data.subsample < data_raw.shape[0]
            LOGGER.info(
                f"Reducing the size of the dataset from {data_raw.shape[0]} to {self.cfg.data.subsample}"
            )
            data_raw = data_raw[: self.cfg.data.subsample, :]
        momentum = data_raw[:, :-1]
        self.momentum = momentum.reshape(momentum.shape[0], momentum.shape[1] // 4, 4)
        self.amplitude = data_raw[:, [-1]]

        # mass regulator
        if self.cfg.data.mass_reg is not None:
            mass = get_mass(self.dataset, self.cfg.data.mass_reg)
            mass = torch.tensor(mass, dtype=self.dtype).unsqueeze(0)
            self.momentum[..., 0] = (self.momentum[..., 1:] ** 2).sum(
                dim=-1
            ) + self.cfg.data.mass_reg**2

        # prepare momenta
        if self.cfg.data.prepare == "align":
            # momenta are already aligned with the beam
            pass
        else:
            if self.cfg.data.prepare == "xyrotation":
                trafo = rand_xyrotation(self.momentum.shape[:-2])
            elif self.cfg.data.prepare == "rotation":
                trafo = rand_rotation(self.momentum.shape[:-2])
            elif self.cfg.data.prepare == "boost":
                trafo = rand_boost(self.momentum.shape[:-2])
            elif self.cfg.data.prepare == "lorentz":
                trafo = rand_lorentz(self.momentum.shape[:-2])
            else:
                raise ValueError(
                    f"cfg.data.prepare={self.cfg.data.prepare} not implemented"
                )
            self.momentum = torch.einsum("...ij,...kj->...ki", trafo, self.momentum)

        # preprocess data
        self.amplitude_prepd, self.amp_mean, self.amp_std = preprocess_amplitude(
            self.amplitude
        )
        self.momentum_prepd = self.momentum / self.momentum.std()
        if self.cfg.data.standardize:
            self.model.init_standardization(self.momentum_prepd)
            self.model.to(device=self.device, dtype=self.dtype)

    def _init_dataloader(self):
        assert sum(self.cfg.data.train_test_val) <= 1

        splits = (
            np.round(np.array(self.cfg.data.train_test_val) * self.amplitude.shape[0])
            .astype("int")
            .tolist()
        )
        trn_amp, tst_amp, val_amp = torch.split(
            self.amplitude_prepd[: sum(splits)], splits, dim=0
        )
        trn_mom, tst_mom, val_mom = torch.split(
            self.momentum_prepd[: sum(splits)], splits, dim=0
        )

        trn_set = torch.utils.data.TensorDataset(trn_amp, trn_mom)
        tst_set = torch.utils.data.TensorDataset(tst_amp, tst_mom)
        val_set = torch.utils.data.TensorDataset(val_amp, val_mom)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=trn_set,
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=tst_set,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=val_set,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        LOGGER.info(
            f"Constructed dataloaders with train_test_val={self.cfg.data.train_test_val}, "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

    @torch.no_grad()
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
                        loader_dict[set_label],
                        set_label,
                    )

                self._evaluate_single(
                    loader_dict[set_label],
                    f"{set_label}_noema",
                )

            else:
                self.results[set_label] = self._evaluate_single(
                    loader_dict[set_label],
                    set_label,
                )

    def _evaluate_single(self, loader, title):
        LOGGER.info(f"### Starting to evaluate model on {title} dataset ###")

        # evaluate model
        # note: shuffle does not matter, because we store both truth and prediction
        self.model.eval()
        t0 = time.time()
        amp_truth_prepd, amp_model_prepd = [], []
        for data in loader:
            amp_model, amp_truth, _ = self._call_model(data)
            amp_model, amp_truth = amp_model.squeeze(dim=-1), amp_truth.squeeze(dim=-1)

            amp_truth_prepd.append(amp_truth.cpu())
            amp_model_prepd.append(amp_model.cpu())
        dt = time.time() - t0
        LOGGER.info(
            f"Evaluation time: {dt*1e6/len(loader.dataset):.2f}s for 1M events "
            f"using batchsize {self.cfg.evaluation.batchsize}"
        )
        amp_truth_prepd = torch.cat(amp_truth_prepd, dim=0)
        amp_model_prepd = torch.cat(amp_model_prepd, dim=0)

        # MSE over preprocessed amplitudes
        mse_prepd = torch.mean((amp_model_prepd - amp_truth_prepd) ** 2)
        LOGGER.info(f"MSE on {title} dataset: {mse_prepd:.4e}")

        # undo preprocessing
        amp_truth = undo_preprocess_amplitude(
            amp_truth_prepd, self.amp_mean, self.amp_std
        )
        amp_model = undo_preprocess_amplitude(
            amp_model_prepd, self.amp_mean, self.amp_std
        )

        # MSE over raw amplitudes
        mse_raw = torch.mean((amp_model - amp_truth) ** 2)

        if self.cfg.use_mlflow:
            log_dict = {
                f"eval.{title}.mse_prepd": mse_prepd,
                f"eval.{title}.mse_raw": mse_raw,
            }
            for key, value in log_dict.items():
                log_mlflow(key, value)

        results = {
            "raw": {
                "truth": amp_truth.numpy(),
                "prediction": amp_model.numpy(),
                "mse": mse_raw,
            },
            "prepd": {
                "truth": amp_truth_prepd.numpy(),
                "prediction": amp_model_prepd.numpy(),
                "mse": mse_prepd,
            },
        }
        return results

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path)
        model_title = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        title = f"{MODEL_TITLE[model_title]} ({DATASET_TITLE[self.dataset]})"
        LOGGER.info(f"Creating plots in {plot_path}")

        plot_dict = {}
        if self.cfg.evaluate and ("test" in self.cfg.evaluation.eval_set):
            plot_dict["results_test"] = self.results["test"]
            plot_dict["results_train"] = self.results["train"]
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
            plot_dict["val_metrics"] = self.val_metrics
            plot_dict["grad_norm"] = self.grad_norm_train
            plot_dict["grad_norm_lframes"] = self.grad_norm_lframes
            plot_dict["grad_norm_net"] = self.grad_norm_net
            for key, value in self.train_metrics.items():
                plot_dict[key] = value
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    def _init_loss(self):
        self.loss = torch.nn.MSELoss()

    def _batch_loss(self, data):
        amp_pred, amp_truth, tracker = self._call_model(data)
        loss = self.loss(amp_truth, amp_pred)

        metrics = tracker
        return loss, metrics

    def _call_model(self, data):
        amplitude, momentum = data
        amplitude, momentum = amplitude.to(self.device), momentum.to(self.device)
        amplitude_model, tracker = self.model(momentum)
        return amplitude_model, amplitude, tracker

    def _init_metrics(self):
        return {"reg_collinear": [], "reg_coplanar": [], "reg_lightlike": []}
