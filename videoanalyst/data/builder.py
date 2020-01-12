# -*- coding: utf-8 -*

from typing import Dict

from yacs.config import CfgNode

import torch
from torch.utils.data import Dataset, DataLoader

# from .sampler import builder as sampler_builder
# from .transformer import builder as transformer_builder
# from .target import builder as target_builder
from .dataloader import AdaptorDataset

def build(task: str, cfg: CfgNode) -> DataLoader:
    r"""
    Arguments
    ---------
    task: str
        task name (track|vos)
    cfg: CfgNode
        node name: data
    """
    # assert task in TASK_SAMPLERS, "invalid task name"

    if task == "track":
        py_dataset = AdaptorDataset(dict(task=task, cfg=cfg), 
                                    num_epochs=cfg.num_epochs, 
                                    nr_image_per_epoch=cfg.nr_image_per_epoch)

        dataloader = DataLoader(py_dataset, batch_size=cfg.minibatch, shuffle=False,
                                pin_memory=True,
                                num_workers=cfg.num_workers, drop_last=True)

    return iter(dataloader)


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
        cfg["sampler"] = sampler_builder.get_config()[task]
        cfg["transformer"] = transformer_builder.get_config()[task]
        cfg["target"] = target_builder.get_config()[task]

    return cfg_dict
