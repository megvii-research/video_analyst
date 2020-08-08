# -*- coding: utf-8 -*
from typing import Dict

from ..monitor_base import TRACK_MONITORS, VOS_MONITORS, MonitorBase


@TRACK_MONITORS.register
@VOS_MONITORS.register
class Monitor(MonitorBase):
    r"""

    Hyper-parameters
    ----------------
    """

    default_hyper_params = dict()

    def __init__(self, ):
        r"""
        Arguments
        ---------
        """
        super(Monitor, self).__init__()

    def init(self, engine_state: Dict):
        super(Monitor, self).init(engine_state)

    def update(self, engine_data: Dict):
        pass
