from typing import Dict, List, Tuple
import numpy as np
import cv2

from yacs.config import CfgNode

from videoanalyst.evaluation.got_benchmark.datasets import got10k
from videoanalyst.data.sampler.sampler_base import TRACK_SAMPLERS, SamplerBase

@TRACK_SAMPLERS.register()
class TrackPairSampler(SamplerBse):
    r"""
    Tracking data sampler

    Hyper-parameters
    ----------------
    """
    default_hyper_params = Dict(
        negative_pair_ratio=0,
    )

    def __init__(self, datsets: List[DatasetBase]=[], seed: int=0, filter=None) -> None:
        super().__init__(cfg, datasets, seed=seed)
        if filter is None:
            self.filter = lambda x: False
        else:
            self.filter = filter

        self._state["ratios"] = [d._hyper_params["ratio"] for d in self.datasets]
        self._state["max_diffs"] = [d._hyper_params["max_diff"] for d in self.datasets]

    def __next__(self) -> Dict:
        is_negative_pair = (self._state["rng"].rand() < self._hyper_params["negative_pair_ratio"]) 

        data1 = data2 = None
        while self.filter(data1) or self.filter(data1):
            if is_negative_pair:
                data1 = self._sample_track_frame()
                data2 = self._sample_track_frame()
            else:
                data1, data2 = self._sample_track_pair()
        
        sampled_data = Dict(
            data1=data1,
            data2=data2,
            is_negative_pair=is_negative_pair,
            )

        return sampled_data

    def _sample_track_pair(self) -> Tuple(Dict, Dict):
        dataset_idx, dataset = self._sample_dataset()
        sequence_data = self._sample_sequence_from_dataset(dataset)
        data1, data2 = self._sample_track_pair_from_sequence_data(
            sequence_data, self._state["max_diffs"][dataset_idx])

        return data1, data2

    def _sample_track_frame(self) -> Dict:
        dataset_idx, dataset = self._sample_dataset()
        sequence_data = self._sample_sequence_from_dataset(dataset)
        data_frame = self._sample_track_frame_from_sequence(sequence_data)

        return datas_frame

    def _sample_dataset(self):
        r"""
        Returns
        -------
        int
            sampled dataset index
        DatasetBase
            sampled dataset
        """
        dataset_ratios = self._cfg.ratios
        dataset_idx = rng.choice(len(dataset_list), p=dataset_ratios)
        dataset = dataset_list[dataset_idx]

        return dataset_idx, dataset
    
    def _sample_sequence_from_dataset(self, dataset: DatasetBase) -> Dict:
        r"""
        """
        len_dataset = len(dataset)
        idx = rng.choice(len(dataset))

        sequence_data = dataset[len_dataset]

        return sequence_data

    def _sample_track_frame_from_sequence(self, sequence_data) -> Dict:
        len_seq = len(List(sequence_data.values)[0])
        idx = rng.choice(len_seq)
        data_frame = {k, v[idx] for k, v in sequence_data.items()}

        return data_frame

    def _sample_track_pair_from_sequence(self, sequence_data, max_diff) -> Tuple(Dict, Dict):
        len_seq = len(List(sequence_data.values)[0])
        idx1, idx2 = self._sample_pair_idx_pair_within_max_diff(len_seq, max_diff)
        data1 = {k, v[idx1] for k, v in sequence_data.items()}
        data2 = {k, v[idx2] for k, v in sequence_data.items()}

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
        idx2_choices = List(range(idx1-max_diff, L)) + \
                    List(range(L+1, idx1+max_diff+1))
        idx2_choices = List(
            set(idx2_choices).intersection(set(range(L)))
        )
        idx2 = rng.choice(idx2_choices)
        return int(idx1), int(idx2)
