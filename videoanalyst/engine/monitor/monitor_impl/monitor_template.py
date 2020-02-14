# -*- coding: utf-8 -*
import itertools
import logging

import cv2
import numpy as np
from tqdm import tqdm

import torch

from ..monitor_base import TRACK_MONITORS, MonitorBase

logger = logging.getLogger("global")


@TRACK_MONITORS.register
class Monitor(MonitorBase):
    r"""

    Hyper-parameters
    ----------------
    """

    default_hyper_params = dict()

    def __init__(self, ):
        r"""
        Arguments
        ---------
        """
        super(Monitor, self).__init__()

    def update(self):
        pass
