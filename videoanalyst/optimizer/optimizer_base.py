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

import torch
from torch import nn

from ..dataset.dataset_base import DatasetBase
from videoanalyst.utils import Registry

from .optimizer_impl.utils import build_lr_scheduler, schedule_lr, multiply_lr

TRACK_OPTIMIZERS = Registry('TRACK_OPTIMIZERS')
VOS_OPTIMIZERS = Registry('VOS_OPTIMIZERS')

TASK_OPTIMIZERS = dict(
    track=TRACK_OPTIMIZERS,
    vos=VOS_OPTIMIZERS,
)

class OptimizerBase:
    __metaclass__ = ABCMeta

    r"""
    base class for Sampler. Reponsible for sampling from multiple datasets and forming training pair / sequence.

    Define your hyper-parameters here in your sub-class.
    """
    default_hyper_params = dict()

    def __init__(self, cfg: CfgNode) -> None:
        r"""
        Dataset Sampler, reponsible for sampling from different dataset

        Arguments
        ---------
        model: nn.Module
            model to registered in optimizer
        """
        self._hyper_params = self.default_hyper_params
        self._state = dict()
        self._cfg = cfg
        self.model = None
        self.optimizer = None
    
    def set_model(self, model: nn.Module):
        r"""
        Arguments
        ---------
        model: nn.Module
            model to registered in optimizer
        """
        self.model = model
        self.optimizer = self.build_optimizer(model)
    
    def build_optimzier(self, model: nn.Module):
        r"""
        an interface to build optimzier
        """

    def build_lr_scheduler(self, cfg: CfgNode):
        r"""
        Arguments
        ---------
        cfg: CfgNode
            node name: lr_scheduler
        """
        self._state["lr_scheduler"] = build_lr_scheduler(cfg)



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
    def schedule_freeze(self, epoch: int, iteration: int) -> None:
        pass


    def schedule_lr(self, epoch: int, iteration: int) -> None:
        r"""
        an interface for optimzier scheduling (e.g. adjust learning rate)
        """
        if "lr_scheduler" in self._state:
            lr = self._state["lr_scheduler"].get_lr(epoch=epoch, iter=iteration)
            schedule_lr(self.optimizer, lr)
            if "lr_multiplier" in self._state:
                multiply_lr(self.optimizer, lr_ratios)
