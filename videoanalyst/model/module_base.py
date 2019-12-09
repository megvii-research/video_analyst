# -*- coding: utf-8 -*
from abc import abstractmethod
from copy import deepcopy

from torch import nn


class ModuleBase(nn.Module):
    r"""
    Module/component base class
    """
    # Define your default hyper-parameters here in your sub-class.
    default_hyper_params = dict()

    def __init__(self):
        super(ModuleBase, self).__init__()
        self._hyper_params = deepcopy(self.default_hyper_params)

    def get_hps(self) -> dict():
        return self._hyper_params

    def set_hps(self, hps: dict()) -> None:
        r"""
        :param hps: dict of hyper-parameters, the keys must in self.__hyper_params
        """
        for key in hps:
            if key not in self._hyper_params:
                raise KeyError
            self._hyper_params[key] = hps[key]

    @abstractmethod
    def update_params(self):
        r"""
        an interface for update params
        """
