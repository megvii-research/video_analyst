# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple
import logging
import numpy as np
import cv2

from yacs.config import CfgNode

from ..template_module_base import TRACK_TEMPLATE_MODULES, VOS_TEMPLATE_MODULES, TemplateModuleBase


@TRACK_TEMPLATE_MODULES.register
@VOS_TEMPLATE_MODULES.register
class TemplateModuleImplementation(TemplateModuleBase):
    r"""
    Template Module Implementation

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict(template_module_hp="", )

    def __init__(self, ) -> None:
        super().__init__()

    def update_params(self) -> None:
        pass
