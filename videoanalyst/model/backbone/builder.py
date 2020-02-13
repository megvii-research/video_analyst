# -*- coding: utf-8 -*
import logging
from typing import Dict

from yacs.config import CfgNode

from .backbone_base import TRACK_BACKBONES, VOS_BACKBONES
from videoanalyst.utils import merge_cfg_into_hps

logger = logging.getLogger(__file__)

TASK_BACKBONES = dict(
    track=TRACK_BACKBONES,
    vos=VOS_BACKBONES,
)


def build(task: str, cfg: CfgNode):
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        buidler configuration

    Returns
    -------
    torch.nn.Module
        module built by builder
    """
    if task in TASK_BACKBONES:
        modules = TASK_BACKBONES[task]
    else:
        logger.error("no backbone for task {}".format(task))
        exit(-1)

    name = cfg.name
    assert name in modules, "backbone {} not registered for {}!".format(
        name, task)
    module = modules[name]()
    hps = module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    module.set_hps(hps)
    module.update_params()
    return module


def get_config() -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {"track": CfgNode(), "vos": CfgNode()}
    for cfg_name, module in zip(["track", "vos"],
                                [TRACK_BACKBONES, VOS_BACKBONES]):
        cfg = cfg_dict[cfg_name]
        cfg["name"] = "unknown"
        for name in module:
            cfg[name] = CfgNode()
            backbone = module[name]
            hps = backbone.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
