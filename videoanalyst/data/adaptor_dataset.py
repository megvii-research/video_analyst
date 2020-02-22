# -*- coding: utf-8 -*
from itertools import chain
from typing import Dict
import logging

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader, Dataset

from videoanalyst.utils.misc import Timer

from .datapipeline import builder as datapipeline_builder

logger = logging.getLogger("global")

_SHARING_STRATETY = "file_system"
if _SHARING_STRATETY in torch.multiprocessing.get_all_sharing_strategies():
    torch.multiprocessing.set_sharing_strategy(_SHARING_STRATETY)


class AdaptorDataset(Dataset):
    _SEED_STEP = 10007
    _SEED_DIVIDER = 1000003
    def __init__(self,
                 kwargs: Dict = dict(),
                 num_epochs=1,
                 nr_image_per_epoch=1):
        self.datapipeline = None
        self.kwargs = kwargs
        self.num_epochs = num_epochs
        self.nr_image_per_epoch = nr_image_per_epoch

    def __getitem__(self, item):
        if self.datapipeline is None:
            seed = (torch.initial_seed() + item*self._SEED_STEP) % self._SEED_DIVIDER
            self.datapipeline = datapipeline_builder.build(**self.kwargs,
                                                           seed=seed)
            logger.info("AdaptorDataset #%d built datapipeline with seed=%d"%(item, seed))

        training_data = next(self.datapipeline)

        return training_data

    def __len__(self):
        return self.nr_image_per_epoch * self.num_epochs
