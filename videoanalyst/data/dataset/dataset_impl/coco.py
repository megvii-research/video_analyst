# import glob

# from .utils import Dataset

# -*- coding: utf-8 -*-
import copy
from loguru import logger
import os
import os.path as osp
import json
from typing import Dict, List
from collections import OrderedDict

import cv2
import numpy as np
from yacs.config import CfgNode

from videoanalyst.data.dataset.dataset_base import TRACK_DATASETS, DatasetBase
from videoanalyst.evaluation.got_benchmark.datasets import ImageNetVID
from videoanalyst.pipeline.utils.bbox import xywh2xyxy


_VALID_SUBSETS = ['train', 'val']


@TRACK_DATASETS.register
class COCODataset(DatasetBase):
    r"""
    COCO dataset helper

    Hyper-parameters
    ----------------
    dataset_root: str
        path to root of the dataset
    subset: str
        dataset split name (train|val)
    ratio: float
        dataset ratio. used by sampler (data.sampler).
    max_diff: int
        maximum difference in index of a pair of sampled frames 
    """
    data_dict = {subset : dict() for subset in _VALID_SUBSETS}
    _DUMMY_ANNO = [[-1, -1, 0, 0]]

    default_hyper_params = dict(
        dataset_root="datasets/COCO",
        subset="train", 
        ratio=1.0,
    )

    def __init__(self) -> None:
        r"""
        Create dataset with config

        Arguments
        ---------
        cfg: CfgNode
            dataset config
        """
        super(COCODataset, self).__init__()
        self._state["dataset"] = None

    def update_params(self):
        r"""
        an interface for update params
        """
        dataset_root = self._hyper_params["dataset_root"]
        self._hyper_params["dataset_root"] = osp.realpath(dataset_root)
        self._ensure_cache()

    def __getitem__(self, item):
        """

        :param item: int, video id
        :return:
            image_files
            annos
            meta (optional)
        """
        # frame_name = 
        subset = self._hyper_params["subset"]
        image_file, anno = COCODataset.data_dict[subset][item]
        if len(anno)<=0:
            anno = self._DUMMY_ANNO
        anno = xywh2xyxy(anno)
        sequence_data = dict(image=[image_file], anno=anno)

        return sequence_data

    def __len__(self):
        subset = self._hyper_params["subset"]
        return len(COCODataset.data_dict[subset])

    def _ensure_cache(self):
        # current_dir = osp.dirname(osp.realpath(__file__))
        dataset_root = self._hyper_params["dataset_root"]
        subset = self._hyper_params["subset"]
        cache_file = osp.join(dataset_root, "cache/coco_%s.json" % subset)

        # dataset_name = type(self).__name__
        if osp.exists(cache_file):
            with open(cache_file, 'r') as f:
                COCODataset.data_dict[subset] = json.load(f)
            logger.info("{}: loaded cache file {}".format(COCODataset.__name__, cache_file))
        else:
            print("cache coco dataset")
            anno_file = osp.join(dataset_root, "annotations/instances_{}.json".format(subset))
            with open(anno_file, 'r') as f:
                annotations = json.load(f)

            images = annotations['images']
            annos = annotations['annotations']

            # reorganize annotations by image
            subset_dir = osp.join(dataset_root, subset)
            subset_dir = osp.realpath(subset_dir)
            data_anno_dict = OrderedDict([
                (image['id'], [osp.join(subset_dir, image['file_name']), []])
                for image in images
            ])

            # iterate over annotation
            for anno in annos:
                # associate by image_id
                image_id = anno['image_id']
                rect = anno['bbox']
                # filter crowd obejct
                if not anno['iscrowd']:
                    data_anno_dict[image_id][1].append(rect) 
            data_anno_list = list(data_anno_dict.values())

            # save internal .json file
            cache_dir = osp.dirname(cache_file)
            if not osp.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(cache_file, 'w') as f:
                json.dump(data_anno_list, f)
            print("COCO dataset: cache dumped at: {}".format(cache_file))
            COCODataset.data_dict[subset] = data_anno_list

