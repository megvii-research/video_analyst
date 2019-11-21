# -*- coding: utf-8 -*
import logging
from typing import Dict

from yacs.config import CfgNode

from videoanalyst.model.loss.loss_base import (TRACK_LOSSES, VOS_LOSSES)

logger = logging.getLogger(__file__)

def build(task:str, cfg:CfgNode):
    if task == "track":
        modules = TRACK_LOSSES
    elif task == "vos":
        modules = VOS_LOSSES
    else:
        logger.error("no loss for task {}".format(task))
        exit(-1)
    
    names = cfg.names
    ret = list()
    for name in names:
        assert name in modules, "loss {} not registered for {}!".format(name, task)
        module = modules[name]()
        hps = module.get_hps()

        for hp_name in hps:
            new_value = cfg[name][hp_name]
            hps[hp_name] = new_value
        module.set_hps(hps)
        module.update_params()
        ret.append(module)
    return ret

def get_config() -> Dict[str, CfgNode]:
    cfg_dict = {"track": CfgNode(), "vos": CfgNode()}
    for cfg_name, module in zip(["track", "vos"],
                                [TRACK_LOSSES, VOS_LOSSES]):
        cfg = cfg_dict[cfg_name]
        cfg["names"] = list()
        for name in module:
            cfg[name] = CfgNode()
            backbone = module[name]
            hps = backbone.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict