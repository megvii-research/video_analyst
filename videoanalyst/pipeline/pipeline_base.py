# -*- coding: utf-8 -*
from copy import deepcopy

from torch import nn


class PipelineBase:
    r"""
    Pipeline base class (e.g. procedure defined for tracker / segmentor / etc.)
    Interface descriptions:
        init(im, state):
        update(im):
    """
    # Define your default hyper-parameters here in your sub-class.
    default_hyper_params = dict()

    def __init__(self, ):
        self._hyper_params = deepcopy(
            self.default_hyper_params)  # mapping-like object
        self._state = dict()  # pipeline state

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

    def init(self, im, state):
        r"""
        an interface for pipeline initialization (e.g. template feature extraction)
        default implementation: record initial state & do nothing
        :param im: initial frame image
        :param state: initial state (usually given by task)
        """
        self._state['state'] = state

    def update(self, im):
        r"""
        an interface for pipeline update
            (e.g. output target bbox for current frame given the frame and previous target bbox)
        default implementation: return previous target state (initial state)
        :param im: current frame
        """
        state = self._state['state']
        return state
