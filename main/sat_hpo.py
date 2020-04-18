# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip

from copy import deepcopy
import argparse
import os.path as osp

from loguru import logger
import yaml
import pandas as pd

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.utils import complete_path_wt_root_in_cfg
from videoanalyst.utils import hpo


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='',
                        type=str,
                        help='experiment configuration')
    parser.add_argument(
        '-hpocfg',
        '--hpo-config',
        default='experiments/siamfcpp/hpo/siamfcpp_SiamFCppTracker-hpo.yaml',
        type=str,
        help='experiment configuration')
    # parser.add_argument('-hpocsv',
    #                     '--hpo-csv',
    #                     default='logs/hpo/hpo.csv',
    #                     type=str,
    #                     help='dumped hpo result')

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
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.test
    task, task_cfg_origin = specify_task(root_cfg)

    # hpo config
    with open(parsed_args.hpo_config, "r") as f:
        hpo_cfg = yaml.safe_load(f)
    hpo_cfg = hpo_cfg["test"]["vos"]
    #_, hpo_cfg = specify_task(hpo_cfg)
    hpo_schedules = hpo.parse_hp_path_and_range(hpo_cfg)

    # results = [hpo.sample_and_update_hps(task_cfg, hpo_schedules) for _ in range(5)]
    # merged_result = hpo.merge_result_dict(results)

    csv_file = osp.join(hpo_cfg["exp_save"],
                        "hpo_{}.csv".format(task_cfg_origin["exp_name"]))

    while True:
        task_cfg = deepcopy(task_cfg_origin)
        hpo_exp_dict = hpo.sample_and_update_hps(task_cfg, hpo_schedules)
        # print(pd.DataFrame(hpo.merge_result_dict(hpo_exp_dict)))

        task_cfg.freeze()
        # build model
        tracker_model  = model_builder.build("track", task_cfg.tracker_model)
        tracker = pipeline_builder.build("track", task_cfg.tracker_pipeline, model=tracker_model)
        segmenter = model_builder.build('vos', task_cfg.segmenter)
        # build pipeline
        pipeline = pipeline_builder.build('vos',
                                          task_cfg.pipeline,
                                          segmenter=segmenter,
                                          tracker=tracker)
        # build tester
        testers = tester_builder(task, task_cfg.tester, "tester", pipeline)
        # start engine
        # for tester in testers:
        tester = testers[0]
        test_result_dict = tester.test()
        hpo_exp_dict["main_performance"] = test_result_dict["main_performance"]
        df = hpo.dump_result_dict(csv_file, hpo_exp_dict)
        df.sort_values(by='main_performance', inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(df.head(10))
        del tracker_model, tracker, segmenter, pipeline, tester
