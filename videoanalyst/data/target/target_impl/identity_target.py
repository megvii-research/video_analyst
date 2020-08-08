# -*- coding: utf-8 -*-

from ..target_base import TRACK_TARGETS, TargetBase


@TRACK_TARGETS.register
class IdentityTarget(TargetBase):
    r"""
    Identity target
    just pass through the data without any modification
    """
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        return data
