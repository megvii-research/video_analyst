# -*- coding: utf-8 -*
import copy
import itertools
import logging
import math
import os
from collections import OrderedDict
from multiprocessing import Process, Queue
from os.path import join

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
