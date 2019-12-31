# -*- coding: utf-8 -*
import logging
from typing import Dict

from yacs.config import CfgNode

from videoanalyst.pipeline.pipeline_base import PipelineBase

from .tester_base import TRACK_TESTERS, VOS_TESTERS

logger = logging.getLogger(__file__)


def build(task: str, cfg: CfgNode, pipeline: PipelineBase):
    """
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        buidler configuration
    pipeline: PipelineBase
        underlying pipeline

    Returns
    -------
    TesterBse
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
    # 此处可以返回多个实验的tester
    for name in names:
        tester = modules[name](cfg, pipeline)
        hps = tester.get_hps()

        # from IPython import embed;embed()
        for hp_name in hps:
            if hp_name in cfg.tester[name]:
                new_value = cfg.tester[name][hp_name]
                hps[hp_name] = new_value
        tester.set_hps(hps)
        tester.update_params()
        testers.append(tester)
    return testers


def get_config() -> Dict[str, CfgNode]:
    """
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
