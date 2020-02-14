# -*- coding: utf-8 -*
from typing import Dict

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader, Dataset

from videoanalyst.utils.misc import Timer

from .datapipeline import builder as datapipeline_builder

torch.multiprocessing.set_sharing_strategy('file_system')




class AdaptorDataset(Dataset):
    default_hyper_params = dict(
        exp_name="",
        exp_save="snapshots",
        num_epochs=10000,
        minibatch=32,
        num_workers=4,
        nr_image_per_epoch=600000,
    )

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
            seed = (torch.initial_seed() + item) % (2**32)
            self.datapipeline = datapipeline_builder.build(**self.kwargs,
                                                           seed=seed)

        training_data = next(self.datapipeline)

        return training_data

    def __len__(self):
        return self.nr_image_per_epoch * self.num_epochs
