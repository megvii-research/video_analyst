# -*- coding: utf-8 -*
import logging
from typing import Dict

from yacs.config import CfgNode

from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS, VOS_TASKMODELS)
from videoanalyst.model.module_base import ModuleBase

logger = logging.getLogger(__file__)


def build(task: str, cfg: CfgNode, backbone: ModuleBase, head: ModuleBase, loss: ModuleBase = None):
    if task == "track":
        task_modules = TRACK_TASKMODELS
    elif task == "vos":
        task_modules = VOS_TASKMODELS
    else:
        logger.error("no task model for task {}".format(task))
        exit(-1)

    if task == "track":
        taskmodel_name = cfg.name
        task_module = task_modules[taskmodel_name](backbone, head, loss)
        hps = task_module.get_hps()
        for hp_name in hps:
            new_value = cfg[taskmodel_name][hp_name]
            hps[hp_name] = new_value
        task_module.set_hps(hps)
        task_module.update_params()
        return task_module
    else:
        logger.error("task model {} is not completed".format(task))
        exit(-1)


def get_config() -> Dict[str, CfgNode]:
    cfg_dict = {"track": CfgNode(), "vos": CfgNode()}
    for cfg_name, task_module in zip(["track", "vos"], [TRACK_TASKMODELS, VOS_TASKMODELS]):
        cfg = cfg_dict[cfg_name]
        cfg["name"] = "unknown"
        for name in task_module:
            cfg[name] = CfgNode()
            task_model = task_module[name]
            hps = task_model.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
