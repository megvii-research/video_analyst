# -*- coding: utf-8 -*

from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.utils.misc import load_cfg
from yacs.config import CfgNode as CN


def build_tracker_wt_model(cfg, device):
    """
    Load model and build tracker given configuration
    :param cfg: config file path or loaded CfgNode
    :param device: device on which model will reside
    :return: tracker pipeline
    """
    if not isinstance(cfg, CN):
        cfg = load_cfg(cfg)
    if hasattr(cfg, 'track'):
        cfg = cfg.track
    model = model_builder.build_model('track', cfg.model)
    tracker = pipeline_builder.build_pipeline('track',
                                              cfg.pipeline,
                                              model=model,
                                              device=device)

    return tracker
