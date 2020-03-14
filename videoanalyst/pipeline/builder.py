# -*- coding: utf-8 -*
import logging
from typing import Dict, List

from yacs.config import CfgNode

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.pipeline.pipeline_base import PIPELINES

# from videoanalyst.model.module_base import TrackerBase

logger = logging.getLogger(__file__)


def build(task: str, cfg: CfgNode, model: ModuleBase):
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        task name
    cfg: CfgNode
        buidler configuration
    model: ModuleBase
        model instance

    Returns
    -------
    torch.nn.Module
        module built by builder
    """
    assert task in PIPELINES, "no pipeline for task {}".format(task)
    pipelines = PIPELINES[task]
    pipeline_name = cfg.name
    pipeline = pipelines[pipeline_name](model)
    hps = pipeline.get_hps()

    for hp_name in hps:
        if hp_name in cfg[pipeline_name]:
            new_value = cfg[pipeline_name][hp_name]
            hps[hp_name] = new_value
    pipeline.set_hps(hps)
    pipeline.update_params()

    return pipeline


def get_config(task_list: List) -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {name: CfgNode() for name in task_list}
    for cfg_name, task_module in PIPELINES.items():
        cfg = cfg_dict[cfg_name]
        cfg["name"] = "unknown"
        for name in task_module:
            cfg[name] = CfgNode()
            task_model = task_module[name]
            hps = task_model.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
