# -*- coding: utf-8 -*

from typing import Dict

from yacs.config import CfgNode

from .tracker import builder as tracker_builder

# from .segmenter import builder as segmenter_builder


def build_pipeline(task: str, cfg: CfgNode, **kwargs):
    if task == "track":
        track_pipeline = tracker_builder.build(cfg, **kwargs)
        return track_pipeline
    else:
        print("model for task {} is not complted".format(task))
        exit(-1)


def get_config() -> Dict[str, CfgNode]:

    cfg_dict = {"track": CfgNode()}

    for task in cfg_dict:
        cfg = cfg_dict[task]
        cfg["tracker"] = tracker_builder.get_config()['track']
    return cfg_dict
