# -*- coding: utf-8 -*
import itertools
from collections import Mapping
import logging
from typing import Dict
import os.path as osp

import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from ..monitor_base import TRACK_MONITORS, VOS_MONITORS, MonitorBase
from videoanalyst.utils import ensure_dir

logger = logging.getLogger("global")


@TRACK_MONITORS.register
@VOS_MONITORS.register
class TensorboardLogger(MonitorBase):
    r"""Log training info to tensorboard for better visualization

    Hyper-parameters
    ----------------
    exp_name : str
        experiment name
    exp_save : str
        directory to save snapshots
    log_dir : str
        places to save tensorboard file
        will be updated in update_params
        EXP_SAVE/EXP_NAME/logs/tensorboard 
    """

    default_hyper_params = dict(
        exp_name="",
        exp_save="",
        log_dir="",
    )

    def __init__(self, ):
        r"""
        Arguments
        ---------
        """
        super(TensorboardLogger, self).__init__()
        self._state["writer"] = None

    def update_params(self):
        self._hyper_params["log_dir"] = osp.join(
            self._hyper_params["exp_save"],
            self._hyper_params["exp_name"],
            "logs/tensorboard",)

    def init(self, engine_state: Dict):
        super(TensorboardLogger, self).init(engine_state)
        log_dir = self._hyper_params["log_dir"]
        ensure_dir(log_dir)
        self._state["writer"] = SummaryWriter(
            log_dir=log_dir,
            purge_step=True,
            filename_suffix="",)

    def update(self, engine_data: Dict):
        # from engine state calculate global step
        engine_state = self._state["engine_state"]
        max_iteration = engine_state["max_iteration"]
        epoch = engine_state["epoch"]
        iteration = engine_state["iteration"]
        global_step = iteration + epoch*max_iteration

        writer = self._state["writer"]

        # traverse engine_data and put to scalar
        self._add_scalar_recursively(writer, engine_data, "", global_step)

    def _add_scalar_recursively(self, writer: SummaryWriter, o, prefix: str, global_step: int):
        """Recursively add scalar from mapping-like o: tag1/tag2/tag3/...
        
        Parameters
        ----------
        writer : SummaryWriter
            writer
        o : mapping-like or scalar
            [description]
        prefix : str
            tag prefix, tag is the name to be passed into writer
        global_step : int
            global step counter
        """
        if isinstance(o, Mapping):
            for k in o:
                if len(prefix) > 0:
                    tag = "%s/%s"%(prefix, k)
                else:
                    tag = k
                self._add_scalar_recursively(writer, o[k], tag, global_step)
        else:
            writer.add_scalar(prefix, o, global_step=global_step)
