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

from ..process_base import TRACK_PROCESSES, ProcessBase
# from videoanalyst.utils import ensure_dir

logger = logging.getLogger("global")


@TRACK_PROCESSES.register
class Process(ProcessBase):
    r"""
    Trainer to test the vot dataset, the result is saved as follows
    exp_dir/logs/$dataset_name$/$tracker_name$/baseline
                                    |-$video_name$/ floder of result files        
                                    |-eval_result.csv evaluation result file

    Hyper-parameters
    ----------------
    """

    default_hyper_params = dict(
    )

    def __init__(self,):
        r"""
        Arguments
        ---------
        """
        super(ProcessBase, self).__init__()

    def update(self):
        pass