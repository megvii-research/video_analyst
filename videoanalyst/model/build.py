# -*- coding: utf-8 -*

from functools import partial
from typing import Dict, List

from megskull.network import NetworkBuilderBase
from yacs.config import CfgNode

from .architecture import builder as archi_builder
from .backbone import builder as backbone_builder
from .loss import builder as loss_builder
