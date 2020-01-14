# -*- coding: utf-8 -*-

from typing import Dict, List

from yacs.config import CfgNode

import torch
from torch import nn

from .optimizer_base import TASK_OPTIMIZERS, OptimizerBase

from videoanalyst.utils import merge_cfg_into_hps

def build(task: str, cfg: CfgNode, model: nn.Module) -> OptimizerBase:
    r"""
    Arguments
    ---------
    task: str
        task name (track|vos)
    cfg: CfgNode
        node name: optimizer
    """
    assert task in TASK_OPTIMIZERS, "invalid task name"
    MODULES = TASK_OPTIMIZERS[task]
    name = cfg.name
    module = MODULES[name](cfg)


    hps = module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    module.set_hps(hps)
    module.update_params()

    if "freeze_scheduler" in cfg: 
        module.build_freeze_scheduler(cfg.freeze_scheduler)
    if "lr_scheduler" in cfg:
        module.build_lr_scheduler(cfg.lr_scheduler)
    module.set_model(model)
    module.build_optimizer()

    return module


def get_config() -> Dict[str, CfgNode]:
    cfg_dict = {name: CfgNode() for name in TASK_SAMPLERS.keys()}

    for cfg_name, modules in TASK_SAMPLERS.items():
        cfg = cfg_dict[cfg_name]
        cfg["names"] = []

        for name in modules:
            cfg[name] = CfgNode()
            module = modules[name]
            hps = module.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]

    return cfg_dict
