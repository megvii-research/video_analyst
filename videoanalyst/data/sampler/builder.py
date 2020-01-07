# -*- coding: utf-8 -*-

from typing import Dict, List

from yacs.config import CfgNode

from .sampler_base import TASK_SAMPLERS, DataSetBase
from ..dataset.builder import build as build_dataset
from ..filter.builder import build as build_filter

def build(task: str, cfg: CfgNode, seed: int=0) -> DataSetBase:
    r"""
    Arguments
    ---------
    task: str
        task name (track|vos)
    cfg: CfgNode
        node name: data
    seed: int
        seed for rng initialization
    """
    assert task in task_datasets, "invalid task name"
    MODULES = TASK_SAMPLERS[task]

    dataset_cfg = cfg.dataset
    datasets = build_dataset(task, dataset_cfg)

    filter_cfg = getattr(cfg, "filter", None)
    filter = build_filter(task, filter_cfg) if filter_cfg is not None else None

    cfg = cfg.sampler
    name = cfg.name
    module = MODULES[name](datasets, seed=seed, filter)

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
