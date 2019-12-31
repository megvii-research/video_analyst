# -*- coding: utf-8 -*
import logging
from typing import Dict

from yacs.config import CfgNode

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_head.taskhead_base import TRACK_HEADS, VOS_HEADS

logger = logging.getLogger(__file__)


def build(task: str, cfg: CfgNode):
    """
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        buidler configuration

    Returns
    -------
    torch.nn.Module
        module built by builder
    """
    if task == "track":
        head_modules = TRACK_HEADS
    elif task == "vos":
        head_modules = VOS_HEADS
    else:
        logger.error("no task model for task {}".format(task))
        exit(-1)

    head_name = cfg.name
    if task == "track":
        # head settings
        head_module = head_modules[head_name]()
        hps = head_module.get_hps()

        for hp_name in hps:
            if hp_name in cfg[head_name]:
                new_value = cfg[head_name][hp_name]
                hps[hp_name] = new_value
        head_module.set_hps(hps)
        head_module.update_params()

        return head_module
    else:
        logger.error("task model {} is not completed".format(task))
        exit(-1)


def get_config() -> Dict[str, CfgNode]:
    """
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {"track": CfgNode(), "vos": CfgNode()}
    for cfg_name, module in zip(["track", "vos"], [TRACK_HEADS, VOS_HEADS]):
        cfg = cfg_dict[cfg_name]
        cfg["name"] = "unknown"
        for name in module:
            cfg[name] = CfgNode()
            task_model = module[name]
            hps = task_model.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
