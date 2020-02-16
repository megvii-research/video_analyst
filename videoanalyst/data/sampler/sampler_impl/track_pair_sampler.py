# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Tuple

import numpy as np
from yacs.config import CfgNode

from videoanalyst.data import _DATA_LOGGER_NAME
from videoanalyst.data.dataset.dataset_base import DatasetBase
from videoanalyst.evaluation.got_benchmark.datasets import got10k
from videoanalyst.utils import load_image

from ..sampler_base import TRACK_SAMPLERS, SamplerBase
import time

# import cv2

data_logger = logging.getLogger(_DATA_LOGGER_NAME)


@TRACK_SAMPLERS.register
class TrackPairSampler(SamplerBase):
    r"""
    Tracking data sampler

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict(negative_pair_ratio=0.0, )

    def __init__(self,
                 datasets: List[DatasetBase] = [],
                 seed: int = 0,
                 filt=None) -> None:
        super().__init__(datasets, seed=seed)
        if filt is None:
            self.filt = [lambda x: False]
        else:
            self.filt = filt

        self._state["ratios"] = [
            d._hyper_params["ratio"] for d in self.datasets
        ]
        self._state["max_diffs"] = [
            d._hyper_params["max_diff"] for d in self.datasets
        ]

    def __next__(self) -> Dict:
        is_negative_pair = (self._state["rng"].rand() <
                            self._hyper_params["negative_pair_ratio"])
        data1 = data2 = None

        while self.filt(data1) or self.filt(data2):
            if is_negative_pair:
                data1 = self._sample_track_frame()
                data2 = self._sample_track_frame()
            else:
                data1, data2 = self._sample_track_pair()
            data1["image"] = load_image(data1["image"], logger=data_logger)
            data2["image"] = load_image(data2["image"], logger=data_logger)

        sampled_data = dict(
            data1=data1,
            data2=data2,
            is_negative_pair=is_negative_pair,
        )

        return sampled_data

    def _sample_track_pair(self) -> Tuple[Dict, Dict]:
        time_a = time.time()
        dataset_idx, dataset = self._sample_dataset()
        time_b = time.time()
        sequence_data = self._sample_sequence_from_dataset(dataset)
        time_c = time.time()
        data1, data2 = self._sample_track_pair_from_sequence(
            sequence_data, self._state["max_diffs"][dataset_idx])
        time_d = time.time()
        #print("{}, {}, {}".format(time_b-time_a, time_c-time_b, time_d-time_c))

        return data1, data2

    def _sample_track_frame(self) -> Dict:
        dataset_idx, dataset = self._sample_dataset()
        sequence_data = self._sample_sequence_from_dataset(dataset)
        data_frame = self._sample_track_frame_from_sequence(sequence_data)

        return data_frame

    def _sample_dataset(self):
        r"""
        Returns
        -------
        int
            sampled dataset index
        DatasetBase
            sampled dataset
        """
        dataset_ratios = self._state["ratios"]
        rng = self._state["rng"]
        dataset_idx = rng.choice(len(self.datasets), p=dataset_ratios)
        dataset = self.datasets[dataset_idx]

        return dataset_idx, dataset

    def _sample_sequence_from_dataset(self, dataset: DatasetBase) -> Dict:
        r"""
        """
        rng = self._state["rng"]
        len_dataset = len(dataset)
        idx = rng.choice(len(dataset))

        sequence_data = dataset[idx]

        return sequence_data

    def _sample_track_frame_from_sequence(self, sequence_data) -> Dict:
        rng = self._state["rng"]
        len_seq = len(list(sequence_data.values())[0])
        idx = rng.choice(len_seq)
        data_frame = {k: v[idx] for k, v in sequence_data.items()}

        return data_frame

    def _sample_track_pair_from_sequence(self, sequence_data: Dict,
                                         max_diff: int) -> Tuple[Dict, Dict]:
        """sample a pair of frames within max_diff distance
        
        Parameters
        ----------
        sequence_data : List
            sequence data: image= , anno=
        max_diff : int
            maximum difference of indexes between two frames in the  pair
        
        Returns
        -------
        Tuple[Dict, Dict]
            track pair data
            data: image= , anno=
        """
        len_seq = len(list(sequence_data.values())[0])
        idx1, idx2 = self._sample_pair_idx_pair_within_max_diff(
            len_seq, max_diff)
        data1 = {k: v[idx1] for k, v in sequence_data.items()}
        data2 = {k: v[idx2] for k, v in sequence_data.items()}

        return data1, data2

    def _sample_pair_idx_pair_within_max_diff(self, L, max_diff):
        r"""
        Draw a pair of index in range(L) within a given maximum difference

        Arguments
        ---------
        L: int
            difference
        max_diff: int
            difference
        """
        rng = self._state["rng"]
        idx1 = rng.choice(L)
        idx2_choices = list(range(idx1-max_diff, L)) + \
                    list(range(L+1, idx1+max_diff+1))
        idx2_choices = list(set(idx2_choices).intersection(set(range(L))))
        idx2 = rng.choice(idx2_choices)
        return int(idx1), int(idx2)
