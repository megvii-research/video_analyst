# -*- coding: utf-8 -*

from yacs.config import CfgNode

from .tester.builder import build as build_tester
from .trainer.builder import build as build_trainer

TASK_ENGINE_BUILDERS = dict(
    tester=build_tester,
    trainer=build_trainer,
)

def build(task: str, cfg: CfgNode, engine_type: str, dataloader, optimizer):
    """
    Builder function for trainer/tester
    engine_type: trainer or tester
    """
    if engine_type in TASK_ENGINE_BUILDERS:
        engine = TASK_ENGINE_BUILDERS[engine_type](task, cfg, dataloader, optimizer)
        return engine
    else:
        raise ValueError("Invalid engine_type: %s"%engine_type)
