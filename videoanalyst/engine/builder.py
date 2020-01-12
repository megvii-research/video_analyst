# -*- coding: utf-8 -*

from yacs.config import CfgNode

from .tester.builder import build as build_tester


def build(task: str, cfg: CfgNode, engine_type: str, **kwargs):
    """
    Builder function for trainer/tester
    engine_type: trainer or tester
    """
    if engine_type == "tester":
        if task == "track":
            testers = build_tester(task, cfg)
            return testers
