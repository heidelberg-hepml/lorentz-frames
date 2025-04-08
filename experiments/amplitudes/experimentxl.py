import os
import queue
import threading
import random
import torch
from torch.utils.data import IterableDataset

from experiments.amplitudes.utils import (
    load_file,
)
from experiments.logger import LOGGER
from experiments.amplitudes.experiment import AmplitudeExperiment


class AmplitudeXLExperiment(AmplitudeExperiment):
    def init_data(self):
        real_subsample = self.cfg.data.subsample
        self.cfg.data.subsample = None
        super().init_data()
        self.cfg.data.subsample = real_subsample

    def _init_dataloader(self):
        super()._init_dataloader(log=False)  # init val and test dataloaders

        assert (
            self.cfg.data.subsample is None or self.cfg.data.num_train_files == 1
        ), "You should not subsample while using multiple files"

        # overwrite self.train_loader
        get_fname = lambda n: os.path.join(
            self.cfg.data.data_path, f"{self.dataset}_{n}.npy"
        )
        file_paths = [get_fname(n + 1) for n in range(self.cfg.data.num_train_files)]
        trn_set = PrefetchFilesDataset(
            file_paths,
            cfg_data=self.cfg.data,
            dataset=self.dataset,
            amp_mean=self.amp_mean,
            amp_std=self.amp_std,
            mom_std=self.mom_std,
            input_dtype=self.dtype,
            num_prefetch=self.cfg.data.num_prefetch,
        )
        self.train_loader = torch.utils.data.DataLoader(
            dataset=trn_set,
            batch_size=self.cfg.training.batchsize,
            num_workers=0,
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )


class PrefetchFilesDataset(IterableDataset):
    """
    Custom dataset to load files on the fly with this strategy:
    - Prepare num_prefetch files in a seperate thread using the load_file function.
    - The DataLoader can access the prepared events with the __iter__ function.
    - We shuffle randomly between files and between events in a single file.
    - Note: cfg_data.subsample allows to control how many events are loaded from each file.
    """

    def __init__(
        self,
        file_paths,
        cfg_data,
        dataset,
        amp_mean,
        amp_std,
        mom_std,
        input_dtype,
        num_prefetch=2,
        events_per_file=1000000,
    ):
        """
        Parameters
        ----------
        file_paths : list of str
            List of paths to the files to load.
        cfg_data
        dataset : str
        amp_mean : float
        amp_std : float
        mom_std : float
        input_dtype : torch.dtype
        num_prefetch : int
            Number of files to prefetch.
        events_per_file : int
            Number of events to yield from each file.
        """
        super().__init__()
        self.events_per_file = (
            events_per_file if cfg_data.subsample is None else cfg_data.subsample
        )

        # prefetch params
        self.file_paths = file_paths
        self.num_prefetch = num_prefetch
        self.rng = random.Random()
        self._EOF = object()

        # load_file arguments
        self.cfg_data = cfg_data
        self.dataset = dataset
        self.amp_mean = amp_mean
        self.amp_std = amp_std
        self.mom_std = mom_std
        self.input_dtype = input_dtype

    def __len__(self):
        return len(self.file_paths) * self.events_per_file

    def _worker(self, file_queue):
        for i, fpath in enumerate(self.shuffled_files):
            # always use the same initial randomness for each file
            generator = torch.Generator().manual_seed(i)
            amp, mom, _, _, _ = load_file(
                fpath,
                cfg_data=self.cfg_data,
                dataset=self.dataset,
                amp_mean=self.amp_mean,
                amp_std=self.amp_std,
                mom_std=self.mom_std,
                input_dtype=self.input_dtype,
                generator=generator,
            )
            idx = torch.randperm(amp.shape[0])
            amp, mom = amp[idx], mom[idx]
            file_queue.put((amp, mom))

        file_queue.put(self._EOF)

    def __iter__(self):
        self.shuffled_files = list(self.file_paths)
        self.rng.shuffle(self.shuffled_files)

        file_queue = queue.Queue(maxsize=self.num_prefetch)
        worker = threading.Thread(target=self._worker, args=(file_queue,))
        worker.daemon = True  # exit if main thread exits
        worker.start()

        while True:
            data = file_queue.get()
            if data is self._EOF:
                break

            amp, mom = data
            for i in range(amp.shape[0]):
                yield amp[i], mom[i]

        worker.join()  # terminate
