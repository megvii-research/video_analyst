# -*- coding: utf-8 -*-
from typing import Dict

import cv2
import numpy as np
from yacs.config import CfgNode

from ..target_base import TRACK_TARGETS, TargetBase
from .utils import make_densebox_target


@TRACK_TARGETS.register
class SATMaskTarget(TargetBase):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        if data["is_negative_pair"]:
            mask_x = data['data2']['anno']
            data['data2']['anno'] = np.zeros_like(mask_x)
        return data
