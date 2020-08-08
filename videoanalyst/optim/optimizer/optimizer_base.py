# -*- coding: utf-8 -*-

from abc import ABCMeta
from typing import Dict

from loguru import logger
from yacs.config import CfgNode

import torch
from torch import nn

from videoanalyst.utils import Registry

from .optimizer_impl.utils.lr_multiply import build as build_lr_multiplier
from .optimizer_impl.utils.lr_policy import build as build_lr_policy
from .optimizer_impl.utils.lr_policy import schedule_lr

OPTIMIZERS = Registry('OPTIMIZERS')


class OptimizerBase:
    __metaclass__ = ABCMeta
    r"""
    base class for Sampler. Reponsible for sampling from multiple datasets and forming training pair / sequence.

    Define your hyper-parameters here in your sub-class.
    """
    default_hyper_params = dict(
        minibatch=1,
        nr_image_per_epoch=1,
        lr_policy=[],
        lr_multiplier=[],
        amp=False,
    )

    def __init__(self, cfg: CfgNode, model: nn.Module) -> None:
        r"""
        Dataset Sampler, reponsible for sampling from different dataset

        Arguments
        ---------
        cfg: CfgNode
            node name: optimizer

        Internal members
        ----------------
        _model:
            underlying nn.Module
        _optimizer
            underlying optim.optimizer.optimizer_base.OptimizerBase
        _scheduler:
            underlying scheduler
        _param_groups_divider: function
            divide parameter for partial scheduling of learning rate 
            input: nn.Module 
            output: List[Dict], k-v: 'params': nn.Parameter
        
        """
        self._hyper_params = self.default_hyper_params
        self._state = dict()
        self._cfg = cfg
        self._model = model
        self._optimizer = None
        self._grad_modifier = None
        self.grad_scaler = None

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
        # calculate & update iteration number
        self._hyper_params["num_iterations"] = self._hyper_params[
            "nr_image_per_epoch"] // self._hyper_params["minibatch"]
        # lr_policy
        lr_policy_cfg = self._hyper_params["lr_policy"]
        if len(lr_policy_cfg) > 0:
            lr_policy = build_lr_policy(
                lr_policy_cfg, max_iter=self._hyper_params["num_iterations"])
            self._state["lr_policy"] = lr_policy
        # lr_multiplier
        lr_multiplier_cfg = self._hyper_params["lr_multiplier"]
        if len(lr_multiplier_cfg) > 0:
            lr_multiplier = build_lr_multiplier(lr_multiplier_cfg)
            self._state["lr_multiplier"] = lr_multiplier
        if "lr_multiplier" in self._state:
            params = self._state["lr_multiplier"].divide_into_param_groups(
                self._model)
        else:
            params = self._model.parameters()

        self._state["params"] = params

        # mix precision
        if self._hyper_params["amp"]:
            try:
                self.grad_scaler = torch.cuda.amp.GradScaler()
            except:
                logger.error(
                    "mix precision training is only supported from torch >=1.6")
                exit()
            logger.info("enabel auto mix precision training")

    def set_grad_modifier(self, grad_modifier):
        self._grad_modifier = grad_modifier

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        if self.grad_scaler is not None:
            self.grad_scaler.step(self._optimizer)
            self.grad_scaler.update()
        else:
            self._optimizer.step()

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)

    def schedule(self, epoch: int, iteration: int) -> Dict:
        r"""
        an interface for optimizer scheduling (e.g. adjust learning rate)
        self.set_scheduler need to be called during initialization phase
        """
        schedule_info = dict()
        if "lr_policy" in self._state:
            lr = self._state["lr_policy"].get_lr(epoch, iteration)
            schedule_lr(self._optimizer, lr)
            schedule_info["lr"] = lr
        # apply learning rate multiplication
        if "lr_multiplier" in self._state:
            self._state["lr_multiplier"].multiply_lr(self._optimizer)

        return schedule_info

    def modify_grad(self, epoch, iteration=-1):
        if self._grad_modifier is not None:
            self._grad_modifier.modify_grad(self._model, epoch, iteration)
