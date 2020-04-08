from typing import Dict

import cv2
import numpy as np
from yacs.config import CfgNode
from loguru import logger

from videoanalyst.data.utils.filter_box import \
    filter_unreasonable_training_boxes, filter_unreasonable_training_masks

from ..filter_base import TRACK_FILTERS, VOS_FILTERS, FilterBase


@TRACK_FILTERS.register
@VOS_FILTERS.register
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
        target_type="bbox",
    )

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: Dict) -> bool:
        if data is None:
            return True
        im, anno = data["image"], data["anno"]
        if self._hyper_params["target_type"] == "bbox":
            filter_flag = filter_unreasonable_training_boxes(
                im, anno, self._hyper_params)
        elif self._hyper_params["target_type"] == "mask":
            filter_flag = filter_unreasonable_training_masks(
                im, anno, self._hyper_params)
        else:
            logger.error("unspported target type {} in filter".format(self._hyper_params["target_type"]))
            exit()
        return filter_flag
