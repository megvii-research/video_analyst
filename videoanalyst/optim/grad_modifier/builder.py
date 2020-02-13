# -*- coding: utf-8 -*-

from typing import Dict, List

from yacs.config import CfgNode

from videoanalyst.utils import merge_cfg_into_hps

from .grad_modifier_base import TASK_GRAD_MODIFIERS, GradModifierBase


def build(task: str, cfg: CfgNode) -> GradModifierBase:
    r"""
    Arguments
    ---------
    task: str
        task name (track|vos)
    cfg: CfgNode
        node name: scheduler
    seed: int
        seed for rng initialization
    """
    assert task in TASK_GRAD_MODIFIERS, "invalid task name"
    MODULES = TASK_GRAD_MODIFIERS[task]

    name = cfg.name
    module = MODULES[name]()

    hps = module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    module.set_hps(hps)
    module.update_params()

    return module


def get_config() -> Dict[str, CfgNode]:
    cfg_dict = {name: CfgNode() for name in TASK_GRAD_MODIFIERS.keys()}

    for cfg_name, MODULES in TASK_GRAD_MODIFIERS.items():
        cfg = cfg_dict[cfg_name]
        cfg["name"] = ""

        for name in MODULES:
            cfg[name] = CfgNode()
            module = MODULES[name]
            hps = module.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]

    return cfg_dict
