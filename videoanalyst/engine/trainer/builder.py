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
    MODULES = TASK_TRAINERS[task]

    model = model_builder(task, cfg.model)
    dataloader = dataloder_builder(task, cfg.data)
    losses = loss_builder(task, cfg.model.losses)
    optimizer = optimizer_builder(task, cfg.model, model)

    cfg = cfg.trainer
    name = cfg.name
    trainer = MODULES[name](model, dataloader, losses, optimizer)
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
    cfg_dict = {"track": CfgNode(), "vos": CfgNode()}

    for cfg_name, module in zip(["track", "vos"], [TRACK_TESTERS, VOS_TESTERS]):
        cfg = cfg_dict[cfg_name]
        cfg["names"] = []
        for name in module:
            cfg["names"].append(name)
            cfg[name] = CfgNode()
            tester = module[name]
            hps = tester.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
