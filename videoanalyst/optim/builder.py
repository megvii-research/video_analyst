# -*- coding: utf-8 -*
from typing import Dict, List

from yacs.config import CfgNode

import torch
from torch import nn

from .grad_modifier import builder as grad_modifier_builder
from .optimizer import builder as optimizer_builder


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
    if "grad_modifier" in cfg:
        grad_modifier = grad_modifier_builder.build(task, cfg.grad_modifier)
        optimizer.set_grad_modifier(grad_modifier)

    return optimizer


def get_config(task_list: List) -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {name: CfgNode() for name in task_list}

    for task in cfg_dict:
        cfg = cfg_dict[task]
        cfg["optimizer"] = optimizer_builder.get_config(task_list)[task]
        cfg["grad_modifier"] = grad_modifier_builder.get_config(task_list)[task]

    return cfg_dict
