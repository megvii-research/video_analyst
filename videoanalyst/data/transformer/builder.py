# -*- coding: utf-8 -*-

from typing import Dict, List

from yacs.config import CfgNode

from .transformer_base import TASK_TRANSFORMERS, DataSetBase


def build(task: str, cfg: CfgNode) -> DataSetBase:
    assert task in TASK_TRANSFORMERS, "invalid task name"
    modules = TASK_TRANSFORMERS[task]

    names = cfg.names

    module = modules[name](cfg, transformer)
    hps = module.get_hps()

    for hp_name in hps:
        new_value = cfg[name][hp_name]
        hps[hp_name] = new_value
    module.set_hps(hps)
    module.update_params()

    return module


def get_config() -> Dict[str, CfgNode]:
    cfg_dict = {name: CfgNode() for name in task_datasets.keys()}

    for cfg_name, modules in task_datasets.items():
        cfg = cfg_dict[cfg_name]
        cfg["names"] = []

        for name in modules:
            cfg[name] = CfgNode()
            module = modules[name]
            hps = module.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]

    return cfg_dict
