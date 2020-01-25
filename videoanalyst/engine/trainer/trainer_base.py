# -*- coding: utf-8 -*
from typing import Dict, List, Tuple
from copy import deepcopy

from torch import nn
from torch.utils.data import DataLoader

from videoanalyst.utils import Registry
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.optimizer.optimizer_base import OptimizerBase


TRACK_TRAINERS = Registry('TRACK_TRAINERS')
VOS_TRAINERS= Registry('VOS_TRAINERS')

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
    default_hyper_params = dict()

    def __init__(self, ):
        self._hyper_params = deepcopy(
            self.default_hyper_params)  # mapping-like object
        self._state = dict()  # pipeline state
        self._model = None
        self._dataloader = None
        self._losses = None
        self._optimizer = None
        self._processes = []

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

    def train(self):
        r"""
        an interface to train for one epoch
        """

    def is_completed(self):
        r""""""

    def set_model(self, model: ModuleBase):
        r""""""
        self._model = model

    def set_dataloader(self, dataloader: DataLoader):
        r""""""
        self._dataloader = dataloader

    def set_losses(self, losses: ModuleBase):
        r""""""
        self._losses = losses

    def set_optimizer(self, optimizer: OptimizerBase):
        r""""""
        self._optimizer = optimizer

    def set_processes(self, processes: List):
        self._processes = processes