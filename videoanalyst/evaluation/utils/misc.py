from yacs.config import CfgNode as CN

from videoanalyst.utils.misc import load_cfg
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder


def build_tracker_wt_model(cfg, device):
    """
    Load model and build tracker given configuration
    :param cfg_path: config file path
    :param device: device on which model will reside
    :return:
    """
    if not isinstance(cfg, CN):
        cfg = load_cfg(cfg)
    model = model_builder.build_model('track', cfg.track.model)
    tracker = pipeline_builder.build_pipeline('track',
                                              cfg.track.pipeline,
                                              model=model,
                                              device=device)

    return tracker
