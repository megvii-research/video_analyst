# -*- coding: utf-8 -*-

import json
import pickle
import tarfile
import time
from abc import ABCMeta, abstractmethod
from typing import List, Dict

import cv2 as cv
import nori2 as nori
import numpy as np

from yacs.config import CfgNode

from videoanalyst.utils import Registry

TRACK_DATASETS = Registry()
VOS_DATASETS = Registry()

TASK_DATASETS = Dict(
    track=TRACK_DATASETS,
    vos=VOS_DATASETS,
)


class DatasetBase:
    __metaclass__ = ABCMeta

    r"""
    base class for DataSet.

    Define your hyper-parameters here in your sub-class.
    """
    default_hyper_params = dict()

    def __init__(self) -> None:
        self._hyper_params = self.default_hyper_params
        self._state = dict()

    def get_hps(self) -> dict():
        r"""
        Getter function for hyper-parameters

        Returns
        -------
        dict
            hyper-parameters
        """
        return self._hyper_params

    def set_hps(self, hps: dict()) -> None:
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
    def update_params(self):
        r"""
        an interface for update params
        """

    def __getitem__(self, item: int) -> dict:
        r"""
        An interface to get data item (Index-based Dataset).
        """

    def __next__(self) -> dict:
        r"""
        An interface to get data item (Sampler-based Dataset).
        """

    def __len__(self):
        r"""
        Length of dataset

        Returns
        -------
        int
            length of dataset
            positive integer if Index-based Dataset
            -1 if Sampler-based Dataset 
        """
