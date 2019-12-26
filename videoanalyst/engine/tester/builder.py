# -*- coding: utf-8 -*
import logging
from typing import Dict

from yacs.config import CfgNode

from .tester_base import TRACK_TESTERS, VOS_TESTERS
from videoanalyst.pipeline.pipeline_base import PipelineBase

logger = logging.getLogger(__file__)

def build(task: str, cfg: CfgNode, pipeline:PipelineBase):
    if task == "track":
        modules = TRACK_TESTERS
    elif task == "vos":
        modules = VOS_TESTERS
    else:
        logger.error("no tester for task {}".format(task))
        exit(-1)
    names = cfg.tester.names
    testers = []
    # 此处可以返回多个实验的tester
    for name in names:
        tester = modules[name]()
        hps = tester.get_hps()

        for hp_name in hps:
            if hp_name in cfg[name]:
                new_value = cfg[name][hp_name]
                hps[hp_name] = new_value
        tester.set_hps(hps)
        tester.update_params()
        testers.append(tester)
    return testers

def get_config() -> Dict[str, CfgNode]:
    cfg_dict = {"track": CfgNode(), "vos": CfgNode()}

    for cfg_name, module in zip(["track", "vos"], [TRACK_TESTERS, VOS_TESTERS]):
        cfg = cfg_dict[cfg_name]
        cfg["names"] = []
        for name in module:
            cfg[name] = CfgNode()
            tester = module[name]
            hps = tester.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict


