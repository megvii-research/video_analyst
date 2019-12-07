# -*- coding: utf-8 -*

# from functools import partial
# from typing import Dict, List

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


# def get_config() -> Dict[str, CfgNode]:
#     cfg_dict = {"track": CfgNode(), "vos": CfgNode()}
#
#     for task in cfg_dict:
#         cfg = cfg_dict[task]
#         cfg["backbone"] = backbone_builder.get_config()[task]
#         cfg["losses"] = loss_builder.get_config()[task]
#         cfg["task_model"] = task_builder.get_config()[task]
#         cfg["task_head"] = head_builder.get_config()[task]
#
#     return cfg_dict
