# -*- coding: utf-8 -*-
from typing import Dict

import numpy as np
import cv2
import os.path as osp

from yacs.config import CfgNode

from videoanalyst.evaluation.got_benchmark.datasets import GOT10k
from videoanalyst.data.dataset.dataset_base import TRACK_DATASETS, DatasetBase
from videoanalyst.pipeline.utils.bbox import xywh2xyxy

@TRACK_DATASETS.register
class GOT10kDataset(DatasetBase):
    r"""
    GOT-10k dataset helper

    Hyper-parameters
    ----------------
    dataset_root: str
        path to root of the dataset
    subset: str
        dataset split name (train|val|test)
    """
    default_hyper_params = dict(
        dataset_root="datasets/GOT-10k",
        subset="train",
        ratio=1,
        max_diff=100,
    )
    def __init__(self) -> None:
        r"""
        Create dataset with config

        Arguments
        ---------
        cfg: CfgNode
            dataset config
        """
        super().__init__()
        self._state["dataset"] = None

    def update_params(self):
        r"""
        an interface for update params
        """
        dataset_root = osp.realpath(self._hyper_params["dataset_root"])
        subset = self._hyper_params["subset"]
        self._state["dataset"] = GOT10k(dataset_root, subset=subset)

    def __getitem__(self, item: int) -> Dict:
        img_files, anno = self._state["dataset"][item]

        anno = xywh2xyxy(anno)
        sequence_data = dict(image=img_files, anno=anno)

        return sequence_data

    def __len__(self):
        return len(self._state["dataset"])
