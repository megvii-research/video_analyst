# -*- coding: utf-8 -*

from functools import partial
from typing import Dict, List

from yacs.config import CfgNode

from .task_model import builder as task_builder
from .backbone import builder as backbone_builder
from .loss import builder as loss_builder
from .task_head import builder as head_builder


def build_model(
        task: str,
        cfg: CfgNode,
):
    if task == "track":
        backbone = backbone_builder.build(task, cfg.backbone)
        losses = loss_builder.build(task, cfg.losses)
        head = head_builder.build(task, cfg.task_head)
        task_model = task_builder.build(task, cfg.task_model, backbone, head, losses)
        return task_model
    else:
        print("model for task {} is not complted".format(task))
        exit(-1)

def get_config() -> Dict[str, CfgNode]:
    cfg_dict = {"track": CfgNode(), "vos": CfgNode()}

    for task in cfg_dict:
        cfg = cfg_dict[task]
        cfg["backbone"] = backbone_builder.get_config()[task]
        cfg["losses"] = loss_builder.get_config()[task]
        cfg["task_model"] = task_builder.get_config()[task]
        cfg["task_head"] = head_builder.get_config()[task]

    return cfg_dict