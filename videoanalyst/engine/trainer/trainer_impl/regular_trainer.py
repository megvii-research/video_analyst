# -*- coding: utf-8 -*
import copy
import itertools
import logging
import math
import os
from collections import OrderedDict
from multiprocessing import Process, Queue
from os.path import join

import cv2
import numpy as np
from tqdm import tqdm

import torch

from ..trainer_base import TRACK_TRAINERS, TrainerBase
from videoanalyst.utils import ensure_dir

logger = logging.getLogger("global")


@TRACK_TRAINERS.register
class RegularTrainer(TrainerBase):
    r"""
    Trainer to test the vot dataset, the result is saved as follows
    exp_dir/logs/$dataset_name$/$tracker_name$/baseline
                                    |-$video_name$/ floder of result files        
                                    |-eval_result.csv evaluation result file

    Hyper-parameters
    ----------------
    device_num: int
        number of gpu for test
    vot_data_root: dict
        vot dataset root directory. dict(dataset_name: path_to_root)
    dataset_names: str
        daataset name (VOT2018|VOT2019)
    """

    default_hyper_params = dict(
        device_num=1,

        dataset_names=[],
    )

    def __init__(self, cfg, pipeline):
        r"""
        Crete tester with config and pipeline

        Arguments
        ---------
        cfg: CfgNode
            parent config, (e.g. model / pipeline / tester)
        pipeline: PipelineBase
            pipeline to test
        """
        super(VOTTester, self).__init__(cfg, pipeline)
        self._state['speed'] = -1

    def test(self):
        r"""
        Run test
        """
        # set dir
        self.tracker_name = self._cfg.exp_name
        for dataset_name in self._hyper_params["dataset_names"]:
            self.dataset_name = dataset_name
            # self.tracker_dir = os.path.join(self._cfg.auto.log_dir, self._hyper_params["dataset_name"])
            self.tracker_dir = os.path.join(self._cfg.exp_save,
                                            self.dataset_name)
            self.save_root_dir = os.path.join(self.tracker_dir,
                                              self.tracker_name, "baseline")
            ensure_dir(self.save_root_dir)
            # track videos
            self.run_tracker()
            # evaluation
            self.evaluation()

