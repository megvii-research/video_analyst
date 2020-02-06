# -*- coding: utf-8 -*
import logging
from typing import Dict

from yacs.config import CfgNode

from .tester_base import TRACK_TESTERS, VOS_TESTERS

from videoanalyst.utils import merge_cfg_into_hps

logger = logging.getLogger(__file__)


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
    TesterBase
        tester built by builder
    """
    if task == "track":
        modules = TRACK_TESTERS
    elif task == "vos":
        modules = VOS_TESTERS
    else:
        logger.error("no tester for task {}".format(task))
        exit(-1)
    names = cfg.tester.names
    testers = []
    # tester for multiple experiments
    for name in names:
        tester = modules[name]()
        hps = tester.get_hps()
        hps = merge_cfg_into_hps(cfg.tester[name], hps)
        tester.set_hps(hps)
        tester.update_params()
        testers.append(tester)
    return testers


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
