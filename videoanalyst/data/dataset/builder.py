# -*- coding: utf-8 -*-

from typing import Dict, List

from yacs.config import CfgNode

# from product_net.data.transformer.transfomer_base import TransformerBase

from .dataset_base import TASK_DATASETS, DatasetBase


def build(task: str, cfg: CfgNode) -> DatasetBase:
    r"""
    Arguments
    ---------
    task: str
        task name (track|vos)
    cfg: CfgNode
        node name: dataset
    """
    assert task in TASK_DATASETS, "invalid task name"
    MODULES = TASK_DATASETS[task]

    names = cfg.names
    modules = []
    for name in names:
        module = MODULES[name]()
        hps = module.get_hps()
        for hp_name in hps:
            new_value = cfg[name][hp_name]
            hps[hp_name] = new_value
        module.set_hps(hps)
        module.update_params()
        modules.append(module)
    
    return modules


def get_config() -> Dict[str, CfgNode]:
    cfg_dict = {name: CfgNode() for name in TASK_DATASETS.keys()}

    for cfg_name, modules in TASK_DATASETS.items():
        cfg = cfg_dict[cfg_name]
        cfg["names"] = []

        for name in modules:
            cfg[name] = CfgNode()
            module = modules[name]
            hps = module.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]

    return cfg_dict
