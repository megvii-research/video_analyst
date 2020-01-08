from typing import Dict, List, Tuple
import numpy as np
import cv2

from yacs.config import CfgNode

from ..filter_base import TRACK_FILTERS, FilterBase
from videoanalyst.data.utils.filter_box import filter_unreasonable_training_boxes

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

