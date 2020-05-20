# -*- coding: utf-8 -*-

import cv2
import numpy as np
from yacs.config import CfgNode

from ..target_base import TRACK_TARGETS, TargetBase
from .utils import make_densebox_target


@TRACK_TARGETS.register
class IdentityTarget(TargetBase):
    r"""
    Identity target
    just pass through the data without any modification
    """
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        return data
