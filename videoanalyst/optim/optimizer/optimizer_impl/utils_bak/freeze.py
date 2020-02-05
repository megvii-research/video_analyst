# -*- coding: utf-8 -*

from typing import List, Dict
import re

from yacs.config import CfgNode

from torch import nn

from collections import OrderedDict
import torch

class FreezeStateMonitor:
    """ Monitor the freezing state continuously and print """
    def __init__(self, module:nn.Module, verbose=True):
        """
        :param module: module to be monitored
        :param verbose:
        """
        self.module = module
        self.verbose = verbose

    def __enter__(self, ):
        self.old_freeze_state = OrderedDict([(k, v.requires_grad) for k, v in self.module.named_parameters()])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.new_freeze_state = OrderedDict([(k, v.requires_grad) for k, v in self.module.named_parameters()])
        if self.verbose:
            assert set(list(self.new_freeze_state.keys())) == set(list(self.old_freeze_state.keys()))
            any_change = False
            for k in self.new_freeze_state.keys():
                change = (self.old_freeze_state[k] != self.new_freeze_state[k])
                if change:
                    print(k, "changed:", self.old_freeze_state[k], "->", self.new_freeze_state[k])
                any_change = any_change or change


def dynamic_freeze(module, param_filter=(lambda x: True), requires_grad=False, verbose=False):
    with FreezeStateMonitor(module, verbose=verbose):
        for k, v in module.named_parameters():
            if param_filter(k):
                v.requires_grad = requires_grad


# def apply_freeze_schedule(module, epoch, schedule_list, verbose=True):
#     with FreezeStateMonitor(module, verbose=verbose):
#         for param_filter, requires_grad_cond in schedule_list:
#             dynamic_freeze(module,
#                            param_filter=param_filter,
#                            requires_grad=requires_grad_cond(epoch))

def apply_freeze_schedule(module, epoch, schedules: List[Dict], verbose=True):
    r"""
    Apply dynamic freezing schedule with verbose
    
    Arguments:
    module: nn.Module
        model to be scheduled
    epoch: int
        current epoch
    schedules: List[Dict]
        lsit of schedule
        schedule: Dict
            "regex": regex to filter parameters
            "epoch": epoch where the schedule starts from
            "freezed": freeze or not

    """
    with FreezeStateMonitor(module, verbose=verbose):
        for schedule in schedule_list:
            # param_filter, requires_grad_cond
            param_filter = schedule["filter"]
            requires_grad = ( (epoch >= schedule["epoch"]) !=
                              schedule["freezed"] )  
            dynamic_freeze(module,
                           param_filter=schedule["filter"],
                           requires_grad=requires_grad)

def resolve_freeze_schedule_cfg(cfg: CfgNode) -> Dict:
    r"""
    Resolve yacs configuration object to get schedule dict
    Arguments
    ---------
    cfg: CfgNode
        node name: freeze_scheduler

    Returns
    -------
    Dict
        returned schedule
    """
    schedule = dict()
    schedule["filter"] = lambda s: re.compile(cfg.regex).search(s) is not None
    schedule["epoch"] = cfg.epoch
    schedule["freezed"] = cfg.freezed
    return schedule
    