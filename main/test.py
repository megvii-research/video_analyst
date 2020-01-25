# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip

import argparse
import logging
import os.path as osp

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder

logger = logging.getLogger('global')


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--config',
                        default='',
                        type=str,
                        help='experiment configuration')

    return parser


if __name__ == '__main__':
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    # resolve config
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()

    # build model
    model = model_builder.build_model(task, task_cfg.model)
    # build pipeline
    pipeline = pipeline_builder.build_pipeline('track', task_cfg.pipeline)
    pipeline.set_model(model)

    # build tester
    testers = tester_builder(task, task_cfg, "tester")
    for tester in testers:
        tester.set_pipeline(pipeline)

    # start engine
    for tester in testers:
        tester.test()
