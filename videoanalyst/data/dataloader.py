# -*- coding: utf-8 -*

from typing import Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from .datapipeline.builder import build as build_datapipeline

from videoanalyst.utils.misc import Timer

class PytorchDataset(Dataset):

    def __init__(self, kwargs: Dict = dict(), num_epochs=1, nr_image_per_epoch=1):
        self.datapipeline = None
        self.kwargs = kwargs
        self.num_epochs = num_epochs
        self.nr_image_per_epoch = nr_image_per_epoch

    def __getitem__(self, item):
        if self.datapipeline is None:
            seed = (torch.initial_seed() + item) % (2**32)
            self.datapipeline = build_datapipeline(**self.kwargs, seed=seed)
        
        # with Timer(enable=False):
        training_data = next(self.datapipeline)

        return training_data

    def __len__(self):
        return self.nr_image_per_epoch * self.num_epochs
