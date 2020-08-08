from __future__ import absolute_import

from .dtb70 import DTB70
from .got10k import GOT10k
from .lasot import LaSOT
from .nfs import NfS
from .otb import OTB
from .tcolor128 import TColor128
from .trackingnet import TrackingNet
from .uav123 import UAV123
from .vid import ImageNetVID
from .vot import VOT

__all__ = [
    GOT10k, OTB, VOT, DTB70, TColor128, UAV123, NfS, LaSOT, TrackingNet,
    ImageNetVID
]
