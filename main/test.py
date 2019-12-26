import sys  # isort:skip
from paths import ROOT_PATH  # isort:skip
sys.path.insert(0, ROOT_PATH)  # isort:skip

import argparse
import logging
import os.path as osp

from main.paths import ROOT_CFG
from videoanalyst.config.config import cfg as whole_config
from videoanalyst.config.config import specify_task
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder

logger = logging.getLogger('global')


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--dataset',
                        dest='dataset',
                        default='VOT2018',
                        help='datasets')
    parser.add_argument('--config',
                        default='',
                        type=str,
                        help='experiment configuration')

    parser.add_argument('--result_file',
                        default='search_epoch.txt',
                        type=str,
                        help='save epoch-level result')
    parser.add_argument('--result_csv',
                        default='search_epoch.csv',
                        type=str,
                        help='save epoch-level result into csv')

    return parser


if __name__ == '__main__':
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    whole_config.merge_from_file(exp_cfg_path)
    logger.info("Load experiment config. at: %s" % exp_cfg_path)
    task, config = specify_task(whole_config)
    config.freeze()
    # build model
    model = model_builder.build_model(task, config.model)
    # build pipeline
    pipeline = pipeline_builder.build_pipeline('track',
                                              config.pipeline,
                                              model=model)
    # build tester
    testers = tester_builder(task, config, "tester")
    # start engine
    for tester in testers:
        tester.test()
