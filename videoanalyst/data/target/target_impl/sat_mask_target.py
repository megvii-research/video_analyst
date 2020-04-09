# -*- coding: utf-8 -*-
from typing import Dict

import cv2
import numpy as np
from yacs.config import CfgNode

from ..target_base import VOS_TARGETS, TargetBase
from videoanalyst.data.utils.crop_track_pair import crop_track_pair_for_sat
from videoanalyst.pipeline.utils.bbox import xywh2xyxy


@VOS_TARGETS.register
class SATMaskTarget(TargetBase):
    default_hyper_params = dict(
        track_z_size=127,
        track_x_size = 303,
        seg_x_size = 129,
        seg_x_resize = 257,
        global_fea_input_size = 129,
        context_amount = 1,
        max_scale = 0.3,
        max_shift = 0.4,
        max_scale_temp = 0.1,
        max_shift_temp = 0.1,

    )
    def __init__(self):
        super().__init__()

    def __call__(self, sampled_data):
        data1 = sampled_data["data1"]
        data2 = sampled_data["data2"]
        im_temp, mask_temp = data1["image"], data1["anno"]
        bbox_temp = cv2.boundingRect(mask_temp)
        bbox_temp = xywh2xyxy(bbox_temp)
        im_curr, mask_curr = data2["image"], data2["anno"]
        bbox_curr = cv2.boundingRect(mask_curr)
        bbox_curr = xywh2xyxy(bbox_curr)
        data_dict = crop_track_pair_for_sat(im_temp, bbox_temp, im_curr,
        bbox_curr, config=self._hyper_params,
        mask_tmp=mask_temp, mask_curr=mask_curr)
        if sampled_data["is_negative_pair"]:
            data_dict["seg_mask"] *= 0

        return data_dict
