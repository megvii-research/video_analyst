# -*- coding: utf-8 -*
import logging
from typing import Dict

from yacs.config import CfgNode

from .trainer_base import TASK_TRAINERS, TrainerBase
from videoanalyst.utils.misc import merge_cfg_into_hps

from videoanalyst.data import builder as dataloder_builder
from videoanalyst.optimizer import builder as optimizer_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.model.loss import builder as loss_builder

logger = logging.getLogger(__file__)


def build(task: str, cfg: CfgNode) -> TrainerBase:
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        node name: train

    Returns
    -------
    TrainerBase
        tester built by builder
    """
    assert task in TASK_TRAINERS, "no tester for task {}".format(task)
    MODULE = TASK_TRAINERS[task]

    cfg = cfg.trainer
    name = cfg.name
    trainer = MODULE[name]()
    hps = trainer.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    trainer.set_hps(hps)
    trainer.update_params()

    return trainer


def get_config() -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {name: CfgNode() for name in TASK_TRAINERS.keys()}

    for cfg_name, modules in TASK_TRAINERS.items():
        cfg = cfg_dict[cfg_name]
        cfg["name"] = ""

        for name in modules:
            # cfg["name"].append(name)
            cfg[name] = CfgNode()
            module = modules[name]
            hps = module.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
