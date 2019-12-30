# -*- coding: utf-8 -*
import logging
from typing import Dict

from yacs.config import CfgNode

from .backbone_base import TRACK_BACKBONES, VOS_BACKBONES

logger = logging.getLogger(__file__)


def build(task: str, cfg: CfgNode):
    """
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
    if task == "track":
        modules = TRACK_BACKBONES
    elif task == "vos":
        modules = VOS_BACKBONES
    else:
        logger.error("no backbone for task {}".format(task))
        exit(-1)

    name = cfg.name
    assert name in modules, "backbone {} not registered for {}!".format(
        name, task)
    module = modules[name]()
    hps = module.get_hps()

    for hp_name in hps:
        if hp_name in cfg[name]:
            new_value = cfg[name][hp_name]
            hps[hp_name] = new_value
    module.set_hps(hps)
    module.update_params()
    return module


def get_config() -> Dict[str, CfgNode]:
    """
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
