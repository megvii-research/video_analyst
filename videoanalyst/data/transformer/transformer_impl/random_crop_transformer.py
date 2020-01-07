from typing import Dict, List, Tuple
import numpy as np
import cv2

from yacs.config import CfgNode

from ..transformer_base import TRACK_TRANSFORMERS, TransformerBase
from videoanalyst.data.utils.crop_track_pair import crop_track_pair
from videoanalyst.pipeline.utils.bbox import xywh2xyxy

@TRACK_SAMPLERS.register()
class RandomCropTransformer(TransformerBase):
    r"""
    Cropping training pair with data augmentation (random shift / random scaling)

    Hyper-parameters
    ----------------

    """
    default_hyper_params = Dict(
        context_amount=0.5,
        max_scale=0.3,
        max_shift=0.4,
        max_scale_temp=1e-6,
        max_shift_temp=1e-6,
        z_size=127,
        x_size=303,
    )

    def __init__(self, cfg: CfgNode, seed: int=0) -> None:
        super().__init__(cfg, seed=seed)
        self.set_hps(self._cfg.)
        crop_cfg = CfgNode()
        crop_cfg.
        self._state["crop_cfg"] = crop_cfg
        
    def __call__(self, sampled_data: Dict) -> Dict:
        data1 = sampled_data["data1"]
        data2 = sampled_data["data2"]


        
