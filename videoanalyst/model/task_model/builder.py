# -*- coding: utf-8 -*
import logging
from typing import Dict

from yacs.config import CfgNode

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)
from videoanalyst.utils import merge_cfg_into_hps

logger = logging.getLogger(__file__)

TASK_TASKMODELS = dict(
    track=TRACK_TASKMODELS,
    vos=VOS_TASKMODELS,
)


def build(task: str,
          cfg: CfgNode,
          backbone: ModuleBase,
          head: ModuleBase,
          loss: ModuleBase = None):
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name
    cfg: CfgNode
        buidler configuration
    backbone: torch.nn.Module
        backbone used by task module.
    head: torch.nn.Module
        head network used by task module.
    loss: torch.nn.Module
        criterion module used by task module (for training). None in case other than training.

    Returns
    -------
    torch.nn.Module
        task module built by builder
    """
    if task == "track":
        task_modules = TRACK_TASKMODELS
    elif task == "vos":
        task_modules = VOS_TASKMODELS
    else:
        logger.error("no task model for task {}".format(task))
        exit(-1)

    if task == "track":
        name = cfg.name
        task_module = task_modules[name](backbone, head, loss)
        hps = task_module.get_hps()
        hps = merge_cfg_into_hps(cfg[name], hps)
        task_module.set_hps(hps)
        task_module.update_params()
        return task_module
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
    for cfg_name, task_module in zip(["track", "vos"],
                                     [TRACK_TASKMODELS, VOS_TASKMODELS]):
        cfg = cfg_dict[cfg_name]
        cfg["name"] = "unknown"
        for name in task_module:
            cfg[name] = CfgNode()
            task_model = task_module[name]
            hps = task_model.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
