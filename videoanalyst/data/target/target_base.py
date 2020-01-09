# -*- coding: utf-8 -*-

import json
import pickle
import tarfile
import time
from abc import ABCMeta, abstractmethod
from typing import List, Dict

import cv2 as cv
import numpy as np

from yacs.config import CfgNode

from videoanalyst.utils import Registry

TRACK_TARGETS = Registry()
VOS_TARGETS = Registry()

TASK_TARGETS = dict(
    track=TRACK_TARGETS,
    vos=VOS_TARGETS,
)

class TargetBase:
    __metaclass__ = ABCMeta

    r"""
    base class for Sampler. Reponsible for sampling from multiple datasets and forming training pair / sequence.

    Define your hyper-parameters here in your sub-class.
    """
    default_hyper_params = dict()

    def __init__(self) -> None:
        r"""
        Target, reponsible for generate training target tensor

        Arguments
        ---------
        cfg: CfgNode
            node name target
        """
        self._hyper_params = self.default_hyper_params
        self._state = dict()

    def get_hps(self) -> Dict:
        r"""
        Getter function for hyper-parameters

        Returns
        -------
        Dict
            hyper-parameters
        """
        return self._hyper_params

    def set_hps(self, hps: Dict) -> None:
        r"""
        Set hyper-parameters

        Arguments
        ---------
        hps: Dict
            Dict of hyper-parameters, the keys must in self.__hyper_params__
        """
        for key in hps:
            if key not in self._hyper_params:
                raise KeyError
            self._hyper_params[key] = hps[key]
    def update_params(self) -> None:
        r"""
        an interface for update params
        """

    def __call__(self, Dict) -> Dict:
        r"""
        An interface to mkae target

        Arguments
        ---------
        Dict
            data whose training target will be made
        """
