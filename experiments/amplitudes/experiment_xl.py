from experiments.amplitudes.experiment import AmplitudeExperiment
from experiments.logger import LOGGER


class XLAmplitudeExperiment(AmplitudeExperiment):
    def init_data(self):
        raise NotImplementedError
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
        raise NotImplementedError
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
