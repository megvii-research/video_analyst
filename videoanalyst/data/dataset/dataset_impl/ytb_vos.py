# import glob

# from .utils import Dataset

# -*- coding: utf-8 -*-
import copy
import io
from loguru import logger
import os
import os.path as osp
import json
import pickle
from typing import Dict, List
from collections import OrderedDict
import contextlib
import cv2
import numpy as np
from yacs.config import CfgNode

from videoanalyst.data.dataset.dataset_base import TRACK_DATASETS, VOS_DATASETS, DatasetBase
from videoanalyst.pipeline.utils.bbox import xywh2xyxy


_VALID_SUBSETS = ['train', 'val']


@TRACK_DATASETS.register
class YoutubeVOSDataset(DatasetBase):
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

    default_hyper_params = dict(
        dataset_root="datasets/youtubevos",
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
        super(YoutubeVOSDataset, self).__init__()
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
        subset = self._hyper_params["subset"]
        record = YoutubeVOSDataset.data_dict[subset][item]
        anno = [[anno_file, record['obj_id']] for anno_file in record["annos"]]
        sequence_data = dict(image=record["image_files"], anno=anno)

        return sequence_data

    def __len__(self):
        subset = self._hyper_params["subset"]
        return len(YoutubeVOSDataset.data_dict[subset])

    def _ensure_cache(self):
        # current_dir = osp.dirname(osp.realpath(__file__))
        dataset_root = self._hyper_params["dataset_root"]
        subset = self._hyper_params["subset"]
        image_root = osp.join(dataset_root, subset, "JPEGImages")
        anno_root = osp.join(dataset_root, subset, "Annotations")
        data_anno_list = []
        cache_file = osp.join(dataset_root, "cache/{}.pkl".format(subset))
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as f:
                YoutubeVOSDataset.data_dict[subset] = pickle.load(f)
            logger.info("{}: loaded cache file {}".format(YoutubeVOSDataset.__name__, cache_file))
        else:
            meta_file = osp.join(dataset_root, subset, "meta.json")
            with open(meta_file) as f:
                records = json.load(f)
            records = records["videos"]
            for video_id in records:
                video = records[video_id]
                for obj_id in video["objects"]:
                    record = video['objects'][obj_id]
                    record['image_files'] = [osp.join(image_root, video_id, frame_id+'.jpg') for frame_id in record['frames']]
                    record['annos'] = [osp.join(anno_root, video_id, frame_id+'.png') for frame_id in record['frames']]
                    record['obj_id'] = int(obj_id)
                    data_anno_list.append(record)
            cache_dir = osp.dirname(cache_file)
            if not osp.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(cache_file, 'wb') as f:
                pickle.dump(data_anno_list, f)
            print("Youtube VOS dataset: cache dumped at: {}".format(cache_file))
            YoutubeVOSDataset.data_dict[subset] = data_anno_list