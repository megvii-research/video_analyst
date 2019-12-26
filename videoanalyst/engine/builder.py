# -*- coding: utf-8 -*

from yacs.config import CfgNode

from .tester.builder import build as build_tester

def build(task: str, cfg: CfgNode, engine_type: str, **kwargs):
    """
    Builder function for trainer/tester
    engine_type: trainer or tester
    """
    if task == "track":
        if engine_type == "tester":
            pipeline = kwargs.get("pipeline")
            return build_tester(task, cfg, pipeline)
        
