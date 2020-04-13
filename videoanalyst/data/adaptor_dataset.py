# -*- coding: utf-8 -*
from itertools import chain
from typing import Dict
from loguru import logger

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.dataloader import default_collate

from videoanalyst.utils.misc import Timer

from .datapipeline import builder as datapipeline_builder

"""
TERMINOLOGY
- seed: usually given by a sub-process rank (inside a dataloader)
- ext_seed: usually given by process rank (dataloader id)
"""

# pytorch wrapper for multiprocessing
# https://pytorch.org/docs/stable/multiprocessing.html#strategy-management
_SHARING_STRATETY = "file_system"
if _SHARING_STRATETY in torch.multiprocessing.get_all_sharing_strategies():
    torch.multiprocessing.set_sharing_strategy(_SHARING_STRATETY)

class AdaptorDataset(Dataset):
    # better to be a prime number
    _EXT_SEED_STEP = 30011
    _SEED_STEP = 10007
    _SEED_DIVIDER = 1000003

    def __init__(
            self,
            task,
            cfg,
            num_epochs=1,
            nr_image_per_epoch=1,
            seed: int = 0,
    ):
        self.datapipeline = None
        self.task = task
        self.cfg = cfg
        self.num_epochs = num_epochs
        self.nr_image_per_epoch = nr_image_per_epoch
        self.ext_seed = seed

    def __getitem__(self, item):
        if self.datapipeline is None:
            # build datapipeline with random seed the first time when __getitem__ is called
            # usually, dataset is already spawned (into subprocess) at this point.
            seed = (torch.initial_seed() + item * self._SEED_STEP +
                    self.ext_seed * self._EXT_SEED_STEP) % self._SEED_DIVIDER
            self.datapipeline = datapipeline_builder.build(self.task,
                                                           self.cfg,
                                                           seed=seed)
            logger.info("AdaptorDataset #%d built datapipeline with seed=%d" %
                        (item, seed))

        training_data = self.datapipeline[item]

        return training_data

    def __len__(self):
        return self.nr_image_per_epoch * self.num_epochs


class AdaptorIterableDataset(IterableDataset):
    # better to be a prime number
    _EXT_SEED_STEP = 30011
    _SEED_STEP = 10007
    _SEED_DIVIDER = 1000003

    def __init__(self,
                 task,
                 cfg,
                 num_epochs=1,
                 nr_image_per_epoch=1,
                 batch_size=1,
                 seed=0, ext_seed=0):
        self.task = task
        self.cfg = cfg
        self.num_epochs = num_epochs
        self.nr_image_per_epoch = nr_image_per_epoch
        self.batch_size = batch_size
        self.seed = (torch.initial_seed() + seed * self._SEED_STEP +
                     ext_seed * self._EXT_SEED_STEP) % self._SEED_DIVIDER

    def _build_datapipeline(self,):
        self.datapipeline = datapipeline_builder.build(self.task, self.cfg, seed=self.seed)
        logger.info("Datapipeline built with seed={}" .format(self.seed))

    def get_streams(self):
        """Stream fetcher"""
        # build datapipeline
        self._build_datapipeline()
        # from each stream, fetch batch_size samples via datapipeline 
        return zip(*[iter(self.datapipeline) for _ in range(self.batch_size)])

    def __iter__(self):
        return self.get_streams()

    def __len__(self):
        return self.nr_image_per_epoch * self.num_epochs

    @classmethod
    def split_datasets(cls, task, cfg, num_epochs, nr_image_per_epoch,
                       batch_size, max_workers, seed=0):
        """ Build a list of splited datasets
            length of list = num_workers
            Each datset fetches a portion of a minibatch
        """
        # find num_workers < max_workers so that batch_size % num_worker = 0
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break
        split_size = batch_size // num_workers  # real batch size for each worker
        logger.debug("batch_size: {}".format(batch_size))
        logger.debug("max_workers: {}".format(max_workers))
        logger.debug("batch size of splitted dataset: {:d}".format(split_size))
        return [
            cls(task,
                cfg,
                num_epochs,
                nr_image_per_epoch,
                batch_size=split_size,
                seed=worker_id, ext_seed=seed) for worker_id in range(num_workers)
        ]

class MultiStreamDataloader:
    def __init__(self, datasets, pin_memory=False):
        self.datasets = datasets  # splited dataset
        self.pin_memory = pin_memory

    # def __length__(self):
    def __len__(self):
        return len(self.datasets[0])

    def get_stream_loaders(self):
        """Generators of list of splited datasets"""
        return zip(*[
            DataLoader(dataset, num_workers=1, batch_size=None, pin_memory=self.pin_memory)
            for dataset in self.datasets
        ])

    def __iter__(self):
        """Generator of minibatch"""
        for batch_parts in self.get_stream_loaders():
            # with Timer(name="default_collect", verbose=False):
            data = default_collate(list(chain(*batch_parts)))
            yield data
