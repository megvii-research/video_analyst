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
    r"""LaSOT tester
    
    Hyper-parameters
    ----------------
    device_num: int
        number of gpus. If set to non-positive number, then use cpu
    data_root: str
        path to got-10k root
    subsets: List[str]
        list of subsets name (val|test)
    """
    extra_hyper_params = dict(
        device_num=1,
        data_root="datasets/LaSOT",
        subsets=["test"],  # (train|test|train_test)
    )

    def __init__(self, *args, **kwargs):
        super(LaSOTTester, self).__init__(*args, **kwargs)
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
        self._pipeline.to_device(dev)
        pipeline_tracker = PipelineTracker(tracker_name, self._pipeline)

        for subset in self._hyper_params["subsets"]:
            root_dir = self._hyper_params["data_root"]
            dataset_name = "GOT-Benchmark"
            save_root_dir = osp.join(self._hyper_params["exp_save"],
                                     dataset_name)
            result_dir = osp.join(save_root_dir, "result")
            report_dir = osp.join(save_root_dir, "report")

            experiment = ExperimentLaSOT(root_dir,
                                         subset=subset,
                                         result_dir=result_dir,
                                         report_dir=report_dir)
            experiment.run(pipeline_tracker)
            experiment.report([tracker_name], plot_curves=False)


LaSOTTester.default_hyper_params = copy.deepcopy(
    LaSOTTester.default_hyper_params)
LaSOTTester.default_hyper_params.update(LaSOTTester.extra_hyper_params)
