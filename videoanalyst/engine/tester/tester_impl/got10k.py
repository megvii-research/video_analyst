# -*- coding: utf-8 -*
import copy
import logging
import os.path as osp

from yacs.config import CfgNode

from ..tester_base import TRACK_TESTERS, TesterBase
from .utils.got_benchmark_helper import PipelineTracker
from videoanalyst.evaluation import got_benchmark
from videoanalyst.evaluation.got_benchmark.experiments import ExperimentGOT10k

logger = logging.getLogger("global")

@TRACK_TESTERS.register
class GOT10kTester(TesterBase):
    extra_hyper_params = dict(
        # device_num=1,
        data_root="datasets/GOT-10k",
        subsets=["val"], # (val|test)
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

            experiment = ExperimentGOT10k(root_dir, subset=subset, 
                                          result_dir=result_dir, report_dir=report_dir)
            experiment.run(pipeline_tracker)
            experiment.report([tracker_name], plot_curves=False)


GOT10kTester.default_hyper_params = copy.deepcopy(GOT10kTester.default_hyper_params)
GOT10kTester.default_hyper_params.update(GOT10kTester.extra_hyper_params)
