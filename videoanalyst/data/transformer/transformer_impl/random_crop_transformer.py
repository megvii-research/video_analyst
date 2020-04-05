from typing import Dict

import cv2
import numpy as np
import torch
from yacs.config import CfgNode

from videoanalyst.data.utils.crop_track_pair import crop_track_pair
from videoanalyst.pipeline.utils.bbox import xywh2xyxy

from ..transformer_base import TRACK_TRANSFORMERS, TransformerBase

@TRACK_TRANSFORMERS.register
class RandomCropTransformer(TransformerBase):
    r"""
    Cropping training pair with data augmentation (random shift / random scaling)

    Hyper-parameters
    ----------------

    """
    default_hyper_params = dict(
        context_amount=1,
        max_scale=0.3,
        max_shift=0.4,
        max_scale_temp=0,
        max_shift_temp=0,
        z_size=127,
        x_size=303,
    )

    def __init__(self, seed: int = 0) -> None:
        super(RandomCropTransformer, self).__init__(seed=seed)

    def __call__(self, sampled_data: Dict) -> Dict:
        r"""
        sampled_data: Dict()
            input data
            Dict(data1=Dict(image, anno), data2=Dict(image, anno))
        """
        data1 = sampled_data["data1"]
        data2 = sampled_data["data2"]
        im_temp, bbox_temp = data1["image"], data1["anno"]
        im_curr, bbox_curr = data2["image"], data2["anno"]
        im_z, bbox_z, im_x, bbox_x, _, _ = crop_track_pair(im_temp,
                                                     bbox_temp,
                                                     im_curr,
                                                     bbox_curr,
                                                     config=self._hyper_params,
                                                     rng=self._state["rng"])

        sampled_data["data1"] = dict(image=im_z, anno=bbox_z)
        sampled_data["data2"] = dict(image=im_x, anno=bbox_x)

        return sampled_data

@TRACK_TRANSFORMERS.register
class RandomCropTransformerWithMask(TransformerBase):
    r"""
    Cropping training pair with data augmentation (random shift / random scaling)

    Hyper-parameters
    ----------------

    """
    default_hyper_params = dict(
        context_amount=1,
        max_scale=0.3,
        max_shift=0.4,
        max_scale_temp=0,
        max_shift_temp=0,
        z_size=127,
        x_size=303,
    )

    def __init__(self, seed: int = 0) -> None:
        super(RandomCropTransformerWithMask, self).__init__(seed=seed)

    def __call__(self, sampled_data: Dict) -> Dict:
        r"""
        sampled_data: Dict()
            input data
            Dict(data1=Dict(image, anno), data2=Dict(image, anno))
        """
        data1 = sampled_data["data1"]
        data2 = sampled_data["data2"]
        im_temp, mask_temp = data1["image"], data1["anno"]
        bbox_temp = cv2.boundingRect(mask_temp)
        bbox_temp = xywh2xyxy(bbox_temp)
        im_curr, mask_curr = data2["image"], data2["anno"]
        bbox_curr = cv2.boundingRect(mask_curr)
        bbox_curr = xywh2xyxy(bbox_curr)
        im_z, bbox_z, im_x, bbox_x, mask_z, mask_x = crop_track_pair(im_temp,
                                                     bbox_temp,
                                                     im_curr,
                                                     bbox_curr,
                                                     config=self._hyper_params,
                                                     rng=self._state["rng"],
                                                     mask_tmp=mask_temp,
                                                     mask_curr=mask_curr)
        sampled_data["data1"] = dict(image=im_z, anno=mask_z)
        sampled_data["data2"] = dict(image=im_x, anno=mask_x)

        return sampled_data
