# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip

import argparse
from loguru import logger
import os.path as osp

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.utils import complete_path_wt_root_in_cfg


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
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
    # from IPython import embed;embed()
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    # resolve config
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()

    if task == 'track':
        # build model
        model = model_builder.build(task, task_cfg.model)
        # build pipeline
        pipeline = pipeline_builder.build('track',
                                          task_cfg.pipeline,
                                          model=model)
        # build tester
        testers = tester_builder(task, task_cfg.tester, "tester", pipeline)

    elif task == 'vos':
        # build model
        tracker = model_builder.build("track_vos", task_cfg.tracker)
        segmenter = model_builder.build('vos', task_cfg.segmenter)
        # build pipeline
        pipeline = pipeline_builder.build('vos',
                                          task_cfg.pipeline,
                                          segmenter=segmenter,
                                          tracker=tracker)
        # build tester
        testers = tester_builder(task, task_cfg.tester, "tester", pipeline)

    for tester in testers:
        tester.test()
