# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple
import numpy as np
import cv2

from yacs.config import CfgNode

from videoanalyst.evaluation.got_benchmark.datasets import got10k
from videoanalyst.data.dataset.dataset_base import DatasetBase
from ..scheduler_base import TRACK_SCHEDULERS, VOS_SCHEDULERS, SchedulerBase
from .utils import lr_policy

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
    )
    def __init__(self, ) -> None:
        super().__init__()
        self._lr_policy = None
        self._lr_multiply = None

    def update_params(self) -> None:
        r"""
        Build lr_policy for scheduler
        """
        lr_policy_cfg = self._hyper_params["lr_policy"]
        _lr_policy = lr_policy.build(lr_policy_cfg)
        self._lr_policy = _lr_policy

    def schedule(self, epoch: int, iteration: int = 0) -> Dict:
        # apply learning rate policy
        state = dict()
        if self._lr_policy is not None:
            lr = self._lr_policy.get_lr(epoch, iteration)
            self._optimizer = lr_policy.schedule_lr(self._optimizer, lr)
            state["lr"] = lr
        # apply learning rate multiplication
        if self._lr_multiply is not None:
            pass
        
        return state
