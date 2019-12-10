import sys  # isort:skip
from paths import ROOT_PATH  # isort:skip
sys.path.insert(0, ROOT_PATH)  # isort:skip

import argparse
import logging
import os.path as osp

from main.paths import ROOT_CFG
from videoanalyst.engine import test
from videoanalyst.utils import load_cfg

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

    # common config
    common_cfg_path = ROOT_CFG
    common_cfg = load_cfg(common_cfg_path)
    logger.info("Load common config. at: %s" % common_cfg_path)

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    exp_cfg = load_cfg(exp_cfg_path)
    logger.info("Load common config. at: %s" % exp_cfg_path)

    # retrieve tester by name & run tester

    # Start engine
    # TESTERS[tester_name](parsed_args, common_cfg, exp_cfg)
    test(parsed_args, common_cfg, exp_cfg)
