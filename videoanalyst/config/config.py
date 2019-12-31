# -*- coding: utf-8 -*-
from yacs.config import CfgNode

from videoanalyst.engine.tester.builder import get_config as get_tester_cfg
from videoanalyst.model.builder import get_config as get_model_cfg
from videoanalyst.pipeline.builder import get_config as get_pipeline_cfg

cfg = CfgNode()

task_list = ["track"]

default_str = "unknown"

cfg["task_name"] = default_str

for task in task_list:
    cfg[task] = CfgNode()
    cfg[task]["exp_name"] = default_str
    cfg[task]["exp_save"] = default_str
    cfg[task]["model"] = get_model_cfg()[task]
    cfg[task]["pipeline"] = get_pipeline_cfg()[task]
    cfg[task]["tester"] = get_tester_cfg()[task]


def specify_task(cfg: CfgNode) -> (str, CfgNode):
    r"""
    get task's short name from config, and specify task config

    Args:
        cfg (CfgNode): config
        
    Returns:
        short task name, task-specified cfg
    """
    for task in task_list:
        if cfg[task].exp_name != default_str:
            return task, cfg[task]
    assert False, "unknown task!"
