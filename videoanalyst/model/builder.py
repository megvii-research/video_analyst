# -*- coding: utf-8 -*
from typing import Dict, List

from yacs.config import CfgNode

from .backbone import builder as backbone_builder
from .loss import builder as loss_builder
from .task_head import builder as head_builder
from .task_model import builder as task_builder


def build(
        task: str,
        cfg: CfgNode,
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
    if task == "track":
        backbone = backbone_builder.build(task, cfg.backbone)
        head = head_builder.build(task, cfg.task_head)
        losses = loss_builder.build(task, cfg.losses)
        task_model = task_builder.build(task, cfg.task_model, backbone, head,
                                        losses)
        return task_model
    else:
        print("model for task {} is not complted".format(task))
        exit(-1)


def get_config(task_list: List) -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {task: CfgNode() for task in task_list}

    for task in cfg_dict:
        cfg = cfg_dict[task]
        cfg["backbone"] = backbone_builder.get_config(task_list)[task]
        cfg["losses"] = loss_builder.get_config(task_list)[task]
        cfg["task_model"] = task_builder.get_config(task_list)[task]
        cfg["task_head"] = head_builder.get_config(task_list)[task]

    return cfg_dict
