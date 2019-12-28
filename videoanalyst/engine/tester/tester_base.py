# -*- coding: utf-8 -*
from copy import deepcopy

from torch import nn

from videoanalyst.utils import Registry

TRACK_TESTERS = Registry('TRACK_TESTERS')
VOS_TESTERS = Registry('VOS_TESTERS')


class TesterBase:
    r"""
    Tester base class (e.g. procedure defined for tracker / segmenter / etc.)
    Interface descriptions:
        init(im, state):
        update(im):
    """
    # Define your default hyper-parameters here in your sub-class.
    default_hyper_params = dict()

    def __init__(self, cfg, pipeline):
        self._hyper_params = deepcopy(
            self.default_hyper_params)  # mapping-like object
        self._state = dict()  # pipeline state
        self._pipeline = pipeline
        self._cfg = cfg

    def get_hps(self) -> dict():
        return self._hyper_params

    def set_hps(self, hps: dict()) -> None:
        r"""
        :param hps: dict of hyper-parameters, the keys must in self.__hyper_params
        """
        for key in hps:
            if key not in self._hyper_params:
                raise KeyError
            self._hyper_params[key] = hps[key]

    def update_params(self):
        r"""
        an interface for update params
        """
    def test(self):
        r"""
        an interface to start testing
        """
