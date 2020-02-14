# -*- coding: utf-8 -*
import logging
from typing import Dict

from yacs.config import CfgNode

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.pipeline.tracker.tracker_base import TRACK_PIPELINES

# from videoanalyst.model.module_base import TrackerBase

logger = logging.getLogger(__file__)


def build(cfg: CfgNode, model: ModuleBase):
    r"""
    Builder function.

    Arguments
    ---------
    cfg: CfgNode
        buidler configuration

    Returns
    -------
    torch.nn.Module
        module built by builder
    """
    track_pipelines = TRACK_PIPELINES
    trackpipeline_name = cfg.name
    track_pipeline = track_pipelines[trackpipeline_name](model)
    hps = track_pipeline.get_hps()

    for hp_name in hps:
        if hp_name in cfg[trackpipeline_name]:
            new_value = cfg[trackpipeline_name][hp_name]
            hps[hp_name] = new_value
    track_pipeline.set_hps(hps)
    track_pipeline.update_params()

    return track_pipeline


def get_config() -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {"track": CfgNode()}
    for cfg_name, task_module in zip(["track"], [TRACK_PIPELINES]):
        cfg = cfg_dict[cfg_name]
        cfg["name"] = "unknown"
        for name in task_module:
            cfg[name] = CfgNode()
            task_model = task_module[name]
            hps = task_model.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
