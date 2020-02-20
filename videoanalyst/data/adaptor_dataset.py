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

torch.multiprocessing.set_sharing_strategy('file_system')

class AdaptorDataset(Dataset):
    def __init__(self,
                 kwargs: Dict = dict(),
                 num_epochs=1,
                 nr_image_per_epoch=1):
        self.datapipeline = None
        self.kwargs = kwargs
        self.num_epochs = num_epochs
        self.nr_image_per_epoch = nr_image_per_epoch
        self.max_iter_per_epoch = 0

    def __getitem__(self, item):
        if self.datapipeline is None:
            seed = (torch.initial_seed() + item*3119) % 10007
            self.datapipeline = datapipeline_builder.build(**self.kwargs,
                                                           seed=seed)
            logger.info("AdaptorDataset #%d built datapipeline with seed=%d"%(item, seed))

        training_data = next(self.datapipeline)

        return training_data

    def __len__(self):
        return self.nr_image_per_epoch * self.num_epochs
