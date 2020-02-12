# -*- coding: utf-8 -*
from typing import Dict, List, Tuple
from copy import deepcopy

from torch import nn
from torch.utils.data import DataLoader

from videoanalyst.utils import Registry
from videoanalyst.model.module_base import ModuleBase

TRACK_PROCESSES = Registry('TRACK_PROCESSES')
VOS_PROCESSES = Registry('VOS_PROCESSES')

TASK_PROCESSES = dict(
    track=TRACK_PROCESSES,
    vos=VOS_PROCESSES,
)

class ProcessBase:
    r"""
    Process base class (e.g. visualization / tensorboard / training info logging)
    """
    # Define your default hyper-parameters here in your sub-class.
    default_hyper_params = dict()

    def __init__(self,):
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
    
    def init(self, engine_state: Dict):
        r"""register engine state & initialize process
        """
        self._state["engine_state"] = engine_state

    def update(self, engine_data: Dict):
        """an interface to update with engine_data and execute defined process
        Execution result will be saved in engine_state

        Parameters
        ----------
        engine_state : Dict
            _state attribute of engine
        engine_data : Dict
            data given by engine at each iteration
        """
