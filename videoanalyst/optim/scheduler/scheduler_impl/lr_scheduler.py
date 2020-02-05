# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple
import numpy as np
import cv2

from yacs.config import CfgNode

from videoanalyst.evaluation.got_benchmark.datasets import got10k
from videoanalyst.data.dataset.dataset_base import DatasetBase
from videoanalyst.optim.scheduler.scheduler_base import TRACK_SCHEDULERS, VOS_SCHEDULERS, SchedulerBase
# from ...optimizer.optimizer_base import OptimizerBase
from .utils import lr_policy, lr_multiply

@TRACK_SCHEDULERS.register
@VOS_SCHEDULERS.register
class LRScheduler(SchedulerBase):
    r"""
    Learning rate scheduler, including:
    - learning rate adjusting
    - learning rate multiplying

    Hyper-parameters
    ----------------
    phases: Dict

    """
    default_hyper_params = dict(
        lr_policy=[],
        lr_multiplier=[],
    )
    def __init__(self, ) -> None:
        super().__init__()
        self._lr_policy = None
        self._lr_multiplier = None

    def update_params(self) -> None:
        r"""
        Build lr_policy for scheduler
        """
        # lr_policy
        lr_policy_cfg = self._hyper_params["lr_policy"]
        _lr_policy = lr_policy.build(lr_policy_cfg)
        self._lr_policy = _lr_policy
        # lr_multiplier
        lr_multiplier_cfg = self._hyper_params["lr_multiplier"]
        _lr_multiplier = lr_multiply.build(lr_multiplier_cfg)
        self._lr_multiplier = _lr_multiplier

    def set_optimizer(self, optimizer) -> None:
        super(LRScheduler, self).set_optimizer(optimizer)
        if self._lr_multiplier is not None:
            optimizer._param_groups_divider = self._lr_multiplier.divide_into_param_groups

    def schedule(self, epoch: int, iteration: int = 0) -> Dict:
        # apply learning rate policy
        optimizer = self._optimizer._optimizer

        state = dict()
        if self._lr_policy is not None:
            lr = self._lr_policy.get_lr(epoch, iteration)
            optimizer = lr_policy.schedule_lr(optimizer, lr)
            state["lr"] = lr
        # apply learning rate multiplication
        if self._lr_multiplier is not None:
            optimizer = self._lr_multiplier.multiply_lr(optimizer)
        
        self._optimizer._optimizer = optimizer        
        return state

