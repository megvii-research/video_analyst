
from typing import Dict

import cv2
import numpy as np
import torch
from yacs.config import CfgNode

from videoanalyst.data.utils.crop_track_pair import crop_track_pair
from videoanalyst.pipeline.utils.bbox import xywh2xyxy

from ..transformer_base import TRACK_TRANSFORMERS, TransformerBase
