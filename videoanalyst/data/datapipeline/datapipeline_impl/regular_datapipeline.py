from typing import Dict, List, Tuple
import numpy as np
import cv2

from yacs.config import CfgNode

from ..datapipeline_base import TRACK_DATAPIPELINES, DatapipelineBase
from ...sampler.sampler_base import SamplerBase

from videoanalyst.utils import convert_data_to_dtype

from videoanalyst.evaluation.got_benchmark.datasets import got10k
from videoanalyst.data.dataset.dataset_base import DatasetBase

@TRACK_DATAPIPELINES.register
class RegularDatapipeline(DatapipelineBase):
    r"""
    Tracking data sampler

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict(
    )

    def __init__(self, sampler: SamplerBase, 
                 pipeline: List = [], ) -> None:
        super().__init__()
        self.sampler = sampler
        self.pipeline = pipeline

    def __next__(self) -> Dict:
        r"""
        An interface to load batch data
        """
        sampled_data = next(self.sampler)

        for proc in self.pipeline:
            sampled_data = proc(sampled_data)

        sampled_data = convert_data_to_dtype(sampled_data)

        return sampled_data
