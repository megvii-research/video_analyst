from typing import Dict, List, Tuple
import numpy as np
import cv2

from yacs.config import CfgNode

from videoanalyst.evaluation.got_benchmark.datasets import got10k
from videoanalyst.data.filter.filter_base import TRACK_FILTERS, FilterBase
from videoanalyst.data.utils.filter_box import filter_unreasonable_training_boxes

@TRACK_FILTERS.register()
class TrackPairFilter(SamplerBse):
    r"""
    Tracking data filter

    Hyper-parameters
    ----------------
    """
    default_hyper_params = Dict(
        max_area_rate=0.6,
        min_area_rate=0.001,
        max_ratio=10,
    )

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        
    def __call__(self, data:Dict) -> bool:
        im, bbox = 
        filter_flag = filter_unreasonable_training_boxes()

        return filter_flag

