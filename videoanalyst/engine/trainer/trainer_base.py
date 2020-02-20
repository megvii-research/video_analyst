# -*- coding: utf-8 -*
from copy import deepcopy
from typing import Dict

from torch import nn
from torch.utils.data import DataLoader

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.optim.optimizer.optimizer_base import OptimizerBase
from videoanalyst.utils import Registry

TRACK_TRAINERS = Registry('TRACK_TRAINERS')
VOS_TRAINERS = Registry('VOS_TRAINERS')

TASK_TRAINERS = dict(
    track=TRACK_TRAINERS,
    vos=VOS_TRAINERS,
)


class TrainerBase:
    r"""
    Trainer base class (e.g. procedure defined for tracker / segmenter / etc.)
    Interface descriptions:
    """
    # Define your default hyper-parameters here in your sub-class.
    default_hyper_params = dict(
        exp_name="default_training",
        exp_save="snapshots",
    )

    def __init__(self, optimizer, dataloader, monitors=[]):
        self._hyper_params = deepcopy(
            self.default_hyper_params)  # mapping-like object
        self._state = dict()  # pipeline state
        self._model = optimizer._model
        self._losses = optimizer._model.loss
        self._optimizer = optimizer
        self._monitors = monitors
        self._optimizer.set_max_iternum_per_epoch(
            dataloader.dataset.max_iter_per_epoch)
        self._max_iter_per_epoch = dataloader.dataset.max_iter_per_epoch
        self._dataloader = iter(dataloader)  # get the iterabel data loader

    def max_iter_per_epoch(self):
        return self._max_iter_per_epoch

    def get_hps(self) -> Dict:
        r"""
        Getter function for hyper-parameters

        Returns
        -------
        dict
            hyper-parameters
        """
        return self._hyper_params

    def set_hps(self, hps: Dict) -> None:
        r"""
        Set hyper-parameters

        Arguments
        ---------
        hps: dict
            dict of hyper-parameters, the keys must in self.__hyper_params__
        """
        for key in hps:
            if key not in self._hyper_params:
                raise KeyError
            self._hyper_params[key] = hps[key]

    def update_params(self):
        r"""
        an interface for update params
        """
    def init_train(self):
        r"""
        an interface to process pre-train overhead before training
        """
        for monitor in self._monitors:
            monitor.init(self._state)

    def train(self):
        r"""
        an interface to train for one epoch
        """
    def is_completed(self):
        r""""""
    def set_dataloader(self, dataloader: DataLoader):
        r""""""
        self._dataloader = dataloader

    def set_optimizer(self, optimizer: OptimizerBase):
        r""""""
        self._optimizer = optimizer
