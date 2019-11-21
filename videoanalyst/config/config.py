# -*- coding: utf-8 -*-
from yacs.config import CfgNode

from videoanalyst.model.builder import get_config as get_model_cfg

cfg = CfgNode()

task_list = ["track", "vos"]

for task in task_list:
    cfg[task] = CfgNode()
    cfg[task]["exp_name"] = "unknown"
    cfg[task]["exp_save"] = "unknown"
    #cfg[task]["data"] = get_data_cfg()[task]
    cfg[task]["model"] = get_model_cfg()[task]
    #cfg[task]["optimize"] = get_opt_cfg()

def specify_task(cfg: CfgNode) -> (str, CfgNode):
    r"""
    get task's short name from config, and specify task config
    :param cfg: config
    :return: short task name, task-specified cfg
    """
    for task in task_list:
        if cfg[task].exp_name != "unknown":
            return task, cfg[task]
    assert False, "unknown task!"