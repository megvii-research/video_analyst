# -*- coding: utf-8 -*
import copy
import logging
import os.path as osp

from yacs.config import CfgNode

from ..tester_base import TRACK_TESTERS, TesterBase
from .utils.got_benchmark_helper import PipelineTracker
from videoanalyst.evaluation import got_benchmark
from videoanalyst.evaluation.got_benchmark.experiments import ExperimentLaSOT

logger = logging.getLogger("global")

@TRACK_TESTERS.register
class LaSOTTester(TesterBase):
    extra_hyper_params = dict(
        # device_num=1,
        data_root="datasets/LaSOT",
        subsets=["test"], # (train|test|train_test)
    )
    def __init__(self, ):
        super().__init__()
        # self._experiment = None

    def test(self, ):
        tracker_name = self._hyper_params["exp_name"]
        pipeline_tracker = PipelineTracker(tracker_name,
                                           self._pipeline)

        for subset in self._hyper_params["subsets"]:
            root_dir = self._hyper_params["data_root"]
            dataset_name = "GOT-Benchmark"
            save_root_dir = osp.join(self._hyper_params["exp_save"], dataset_name)
            result_dir = osp.join(save_root_dir, "result")
            report_dir = osp.join(save_root_dir, "report")

            experiment = ExperimentLaSOT(root_dir, subset=subset, 
                                         result_dir=result_dir, report_dir=report_dir)
            experiment.run(pipeline_tracker)
            experiment.report([tracker_name], plot_curves=False)

LaSOTTester.default_hyper_params = copy.deepcopy(LaSOTTester.default_hyper_params)
LaSOTTester.default_hyper_params.update(LaSOTTester.extra_hyper_params)
