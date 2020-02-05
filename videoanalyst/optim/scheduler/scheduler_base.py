# -*- coding: utf-8 -*-
import json
import pickle
import tarfile
import time
from abc import ABCMeta, abstractmethod
from typing import List, Dict

import cv2 as cv
import numpy as np

import torch
from torch import nn
from torch import optim

from yacs.config import CfgNode

from videoanalyst.utils import Registry
# from videoanalyst.optim.optimizer.optimizer_base import OptimizerBase

TRACK_SCHEDULERS = Registry('TRACK_SCHEDULER')
VOS_SCHEDULERS = Registry('VOS_SCHEDULER')

TASK_SCHEDULERS = dict(
    track=TRACK_SCHEDULERS,
    vos=VOS_SCHEDULERS,
)

class SchedulerBase:
    __metaclass__ = ABCMeta

    r"""
    base class for Scheduler. Reponsible for scheduling optimizer (learning rate) during training

    Define your hyper-parameters here in your sub-class.
    """
    default_hyper_params = dict()

    def __init__(self, ) -> None:
        r"""
        Scheduler, reponsible for scheduling optimizer

        Arguments
        ---------
        cfg: CfgNode
            data config, including cfg for datasset / sampler
        
        s: List[DatasetBase]
            collections of datasets
        seed: int
            seed to initialize random number generator
            important while using multi-worker data loader
        """
        self._hyper_params = self.default_hyper_params
        self._state = dict()
        self._model = None
        self._optimizer = None

    def get_hps(self) -> dict:
        r"""
        Getter function for hyper-parameters

        Returns
        -------
        dict
            hyper-parameters
        """
        return self._hyper_params

    def set_hps(self, hps: dict) -> None:
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
            
    def update_params(self) -> None:
        r"""
        an interface for update params
        """
    
    def set_model(self, model: nn.Module) -> None:
        r"""set model for scheduler"""
        self._model = model

    def set_optimizer(self, optimizer) -> None:
        r"""get underlying optimizer and set it for scheduler"""
        self._optimizer = optimizer
        # assert isinstance(self._optimizer, optim.Optimizer)

    def schedule(self, epoch: int, iteration: int) -> Dict:
        r"""
        Schedule the underlying optimizer/model
        
        Parameters
        ----------
        epoch : int
            [description]
        iteration : int
            [description]
        Returns
        -------
        Dict:
            dict containing the schedule state
        """
        