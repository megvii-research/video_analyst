# -*- coding: utf-8 -*

from typing import Dict

from yacs.config import CfgNode

from .tracker import builder as tracker_builder
from .tracker.tracker_base import TRACK_PIPELINES
from videoanalyst.model.module_base import ModuleBase

# from .segmenter import builder as segmenter_builder


def build_pipeline(task: str, cfg: CfgNode, model: ModuleBase):
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
        track_pipeline = tracker_builder.build(cfg, model)
        return track_pipeline
    else:
        print("model for task {} is not complted".format(task))
        exit(-1)


def get_config() -> Dict[str, CfgNode]:
    """
    Get available component list config

    Returns
    -------
    CfgNode
        config with list of available components
    """
    cfg_dict = {"track": CfgNode()}

    for cfg_name, module in zip([
            "track",
    ], [
            TRACK_PIPELINES,
    ]):
        cfg = cfg_dict[cfg_name]
        cfg["name"] = "unknown"
        for name in module:
            cfg[name] = CfgNode()
            tester = module[name]
            hps = tester.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
