from __future__ import absolute_import

from .dtb70 import ExperimentDTB70
from .got10k import ExperimentGOT10k
from .lasot import ExperimentLaSOT
from .nfs import ExperimentNfS
from .otb import ExperimentOTB
from .tcolor128 import ExperimentTColor128
from .trackingnet import ExperimentTrackingNet
from .uav123 import ExperimentUAV123
from .vot import ExperimentVOT

__all__ = [
    ExperimentGOT10k, ExperimentOTB, ExperimentVOT, ExperimentDTB70,
    ExperimentUAV123, ExperimentNfS, ExperimentTColor128, ExperimentLaSOT,
    ExperimentTrackingNet
]
