# -*- coding: utf-8 -*-

import json
import pickle
import tarfile
import time
from abc import ABCMeta, abstractmethod
from typing import List, Dict

import cv2 as cv
import numpy as np

from yacs.config import CfgNode

from ..sampler.sampler_base import SamplerBase
from ..transformer.transformer_base import TransformerBase
from ..target.target_base import TargetBase

from ..dataset.builder import build as build_dataset
from ..filter.builder import build as build_filter

from videoanalyst.utils import Registry
TRACK_DATAPIPELINES = Registry('TRACK_DATAPIPELINES')
VOS_DATAPIPELINES = Registry('VOS_DATAPIPELINES')

TASK_DATAPIPELINES = dict(
    track=TRACK_DATAPIPELINES,
    vos=VOS_DATAPIPELINES,
)

class DatapipelineBase:
    __metaclass__ = ABCMeta

    r"""
    base class for Sampler. Reponsible for sampling from multiple datasets and forming training pair / sequence.

    Define your hyper-parameters here in your sub-class.
    """
    default_hyper_params = dict()

    def __init__(self) -> None:
        r"""
        Data pipeline
        """
        self._hyper_params = self.default_hyper_params
        self._state = dict()

    def get_hps(self) -> dict:
        r"""
        Getter function for hyper-parameters

        Returns
        -------
        dict
            hyper-parameters
        """
        return self._hyper_params

    def set_hps(self, hps: dict) -> None:
        r"""
        Set hyper-parameters

        Arguments
        ---------
        hps: dict
            dict of hyper-parameters, the keys must in self.__hyper_params__
        """
        for key in hps:
            if key not in self._hyper_params:
                raise KeyError
            self._hyper_params[key] = hps[key]
            
    def update_params(self) -> None:
        r"""
        an interface for update params
        """

    def __next__(self) -> Dict:
        r"""
        An interface to load batch data
        """
