# -*- coding: utf-8 -*-
from yacs.config import CfgNode

from ..data.builder import get_config as get_data_cfg
from ..model.builder import get_config as get_model_cfg
from ..optimize.builder import get_confg as get_opt_cfg

cfg = CfgNode()

for task in ["track", "vos"]:
    cfg[task] = CfgNode()
    cfg[task]["exp_name"] = "unknown"
    cfg[task]["exp_save"] = "unknown"
    cfg[task]["data"] = get_data_cfg()[task]
    cfg[task]["model"] = get_model_cfg()[task]
    cfg[task]["optimize"] = get_opt_cfg()
    cfg[task]["manager"] = get_manager_cfg()[task]
