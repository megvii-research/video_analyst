# -*- coding: utf-8 -*-

from typing import Dict, List

from yacs.config import CfgNode

from .datapipeline_base import TASK_DATAPIPELINES, DatapipelineBase

from ..sampler.builder import build as build_sampler
from ..transformer.builder import build as build_transformer
from ..target.builder import build as build_target

from videoanalyst.utils import merge_cfg_into_hps

def build(task: str, cfg: CfgNode, seed: int=0) -> DatapipelineBase:
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
    assert task in TASK_DATAPIPELINES, "invalid task name"
    MODULES = TASK_DATAPIPELINES[task]

    sampler =  build_sampler(task, cfg.sampler, seed=seed)
    transformers = build_transformer(task, cfg.transformer, seed=seed)
    target = build_target(task, cfg.target)

    pipeline = []
    pipeline.extend(transformers)
    pipeline.append(target)

    cfg = cfg.datapipeline
    name = cfg.name
    module = MODULES[name](sampler, pipeline)

    # hps = module.get_hps()
    # hps = merge_cfg_into_hps(cfg[name], hps)
    # module.set_hps(hps)
    # module.update_params()

    return module


def get_config() -> Dict[str, CfgNode]:
    cfg_dict = {name: CfgNode() for name in TASK_DATALOADERS.keys()}

    for cfg_name, modules in TASK_DATALOADERS.items():
        cfg = cfg_dict[cfg_name]
        cfg["names"] = []

        for name in modules:
            cfg[name] = CfgNode()
            module = modules[name]
            hps = module.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]

    return cfg_dict
