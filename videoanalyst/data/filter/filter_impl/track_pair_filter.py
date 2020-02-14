from typing import Dict

import cv2
import numpy as np
from yacs.config import CfgNode

from videoanalyst.data.utils.filter_box import \
    filter_unreasonable_training_boxes

from ..filter_base import TRACK_FILTERS, FilterBase


@TRACK_FILTERS.register
class TrackPairFilter(FilterBase):
    r"""
    Tracking data filter

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict(
        max_area_rate=0.6,
        min_area_rate=0.001,
        max_ratio=10,
    )

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: Dict) -> bool:
        if data is None:
            return True
        im, bbox = data["image"], data["anno"]
        filter_flag = filter_unreasonable_training_boxes(
            im, bbox, self._hyper_params)

        return filter_flag
