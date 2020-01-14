# -*- coding: utf-8 -*
from typing import Dict, List, Tuple
from copy import deepcopy

from torch import nn

from videoanalyst.utils import Registry
from videoanalyst.model.module_base import ModuleBase

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

    # def __init__(self, ):
    def __init__(self, 
                 model: ModuleBase, 
                 dataloader: DataLoader, 
                 losses: ModuleBase, 
                 optimizer: OptimizerBase, 
                 process=[]):
        self._hyper_params = deepcopy(
            self.default_hyper_params)  # mapping-like object
        self._state = dict()  # pipeline state

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
        
    def train(self):
        r"""
        an interface to train for one epoch
        """
