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
from torch.optim.optimizer import Optimizer

from videoanalyst.utils import Registry

from torch.optim.optimizer import Optimizer

from .optimizer_impl.utils.lr_policy import build as build_lr_scheduler, schedule_lr
from .optimizer_impl.utils.lr_multiply import multiply_lr, resolve_lr_multiplier_cfg
from .optimizer_impl.utils.freeze import apply_freeze_schedule, resolve_freeze_schedule_cfg

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
        cfg: CfgNode
            node name: optimizer
        """
        self._hyper_params = self.default_hyper_params
        self._state = dict()
        self._cfg = cfg
        self._model = None
        self._optimizer = None
    
    def set_model(self, model: nn.Module):
        r"""
        Arguments
        ---------
        model: nn.Module
            model to registered in optimizer
        """
        self._model = model

    def build_optimizer(self):
        r"""
        an interface to build optimzier
        """

    def build_freeze_scheduler(self, cfg: CfgNode):
        r"""
        Arguments
        ---------
        cfg: Cfgnode
        """
        schedules = [resolve_freeze_schedule_cfg(cfg[k]) for k in cfg]
        self._state["freeze_schedules"] = schedules

    def build_lr_scheduler(self, cfg: CfgNode):
        r"""
        Arguments
        ---------
        cfg: CfgNode
            node name: lr_scheduler
        """
        self._state["lr_scheduler"] = build_lr_scheduler(cfg)

    # def build_lr_multiplier(self, cfg: CfgNode):
    #     lr_multipliers = [resolve_lr_schedule_cfg(cfg[k]) for k in cfg]
    #     self._state["lr_multipliers"] = lr_multipliers

    # def get_param_group(self):
    #     pass

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
    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        self._optimizer.step()

    def state_dict(self):
        self._optimizer.state_dict()

    def schedule_freeze(self, epoch: int) -> None:
        schedules = self._state.get("freeze_schedules", [])
        apply_freeze_schedule(self._model, epoch, schedules)

    def schedule_lr(self, epoch: int, iteration: int) -> None:
        r"""
        an interface for optimzier scheduling (e.g. adjust learning rate)
        """
        if "lr_scheduler" in self._state:
            lr = self._state["lr_scheduler"].get_lr(epoch=epoch, iter=iteration)
            schedule_lr(self._optimizer, lr)
            if "lr_multiplier" in self._state:
                multiply_lr(self._optimizer, lr_ratios)
    
    # def multiply_lr(self, epoch: int) -> None:
    #     if "lr_multiplier" in self._state:
    #         multiply_lr(self.optimizer, self._state["lr_multiplier"])
