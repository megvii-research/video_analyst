# -*- coding: utf-8 -*
import copy
from loguru import logger
import os.path as osp

from yacs.config import CfgNode

import torch

from videoanalyst.evaluation import got_benchmark
from videoanalyst.evaluation.got_benchmark.experiments import ExperimentTrackingNet

from ..tester_base import TRACK_TESTERS, TesterBase
from .utils.got_benchmark_helper import PipelineTracker


@TRACK_TESTERS.register
class TrackingNetTester(TesterBase):
    r"""TrackingNet tester
    
    Hyper-parameters
    ----------------
    device_num: int
        number of gpus. If set to non-positive number, then use cpu
    data_root: str
        path to TrackingNet root
    subsets: List[str]
        list of subsets name (val|test)
    """
    extra_hyper_params = dict(
        device_num=1,
        data_root="datasets/TrackingNet",
        subsets=["TEST"],  # (val|test)
    )

    def __init__(self, *args, **kwargs):
        super(TrackingNetTester, self).__init__(*args, **kwargs)
        # self._experiment = None

    def update_params(self):
        # set device state
        num_gpu = self._hyper_params["device_num"]
        if num_gpu > 0:
            all_devs = [torch.device("cuda:%d" % i) for i in range(num_gpu)]
        else:
            all_devs = [torch.device("cpu")]
        self._state["all_devs"] = all_devs

    def test(self, ):
        tracker_name = self._hyper_params["exp_name"]
        all_devs = self._state["all_devs"]
        dev = all_devs[0]
        self._pipeline.set_device(dev)
        pipeline_tracker = PipelineTracker(tracker_name, self._pipeline)

        for subset in self._hyper_params["subsets"]:
            root_dir = self._hyper_params["data_root"]
            dataset_name = "GOT-Benchmark"
            save_root_dir = osp.join(self._hyper_params["exp_save"],
                                     dataset_name)
            result_dir = osp.join(save_root_dir, "result")
            report_dir = osp.join(save_root_dir, "report")

            experiment = ExperimentTrackingNet(root_dir,
                                          subset=subset,
                                          result_dir=result_dir,
                                          report_dir=report_dir)
            experiment.run(pipeline_tracker, overwrite_result=True)
            performance = experiment.report([tracker_name], plot_curves=False)

        test_result_dict = dict()
        if performance is not None:
            test_result_dict["main_performance"] = performance[tracker_name][
                "overall"]["ao"]
        else:
            test_result_dict["main_performance"] = -1
        return test_result_dict


TrackingNetTester.default_hyper_params = copy.deepcopy(
    TrackingNetTester.default_hyper_params)
TrackingNetTester.default_hyper_params.update(TrackingNetTester.extra_hyper_params)
