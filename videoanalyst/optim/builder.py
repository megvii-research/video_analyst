# -*- coding: utf-8 -*
from typing import Dict

from yacs.config import CfgNode

import torch
from torch import nn

from .optimizer import builder as optimizer_builder
from .scheduler import builder as scheduler_builder


def build(
        task: str,
        cfg: CfgNode,
        model: nn.Module,
):
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        node name: model

    Returns
    -------
    torch.nn.Module
        module built by builder
    """
    optimizer = optimizer_builder.build(task, cfg.optimizer, model)
    if "scheduler" in cfg:
        scheduler = scheduler_builder.build(task, cfg.scheduler)    
        optimizer.set_scheduler(scheduler)

    return optimizer


def get_config() -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {"track": CfgNode(), "vos": CfgNode()}

    for task in cfg_dict:
        cfg = cfg_dict[task]
        cfg["optimizer"] = optimizer_builder.get_config()[task]
        cfg["scheduler"] = scheduler_builder.get_config()[task]

    return cfg_dict
