# -*- coding: utf-8 -*

from yacs.config import CfgNode as CN

from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.utils.misc import load_cfg


def build_tracker_wt_model(cfg, device):
    r"""
    Load model and build tracker given configuration

    Arguments
    ---------
    cfg: CfgNode
        config file path or loaded CfgNode
    device: torch.device
        device on which model will reside

    Returns
    -------
    PipelineBase
        tracker pipeline
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
