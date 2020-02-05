# -*- coding: utf-8 -*
from typing import List, Dict
from collections import OrderedDict
import json
import re

from yacs.config import CfgNode

import torch
from torch import nn
from torch import optim


def build(cfg: List[str]):
    """Build lr multiplier
    
    Parameters
    ----------
    cfg : List[str]
        list of JSON string, containing multiplier info (e.g. name, regex, ratio, etc.)
    
    Returns
    -------
    LRMultiplier
        multiplier providing following operation: divide_into_param_groups and lr_multiply
    """
    schedule = dict(name=[], regex=[], ratio=[], compiled_regex=[])
    for mult_str in cfg:
        mult_cfg = json.loads(mult_str)
        schedule["name"].append(mult_cfg["name"])
        schedule["regex"].append(mult_cfg["regex"])
        schedule["ratio"].append(mult_cfg["ratio"])
        compiled_regex = re.compile(mult_cfg["regex"])
        schedule["compiled_regex"].append(compiled_regex)
        # schedule["filter"].append(lambda s: re.compile(mult_cfg["regex"]).search(s) is not None)

    multipiler = LRMultiplier(schedule["name"], schedule["compiled_regex"], schedule["ratio"])
    # from IPython import embed;embed()

    return multipiler
    

class LRMultiplier():
    def __init__(self, names: List[str], compiled_regexes: List, ratios: List[float]):
        """multiplier
        
        Parameters
        ----------
        names : List[str]
            name of group
        filters : List
            function for filter parameters by name (e.g. compiled regex with re package)
        ratios : List[float]
            multiplication ratio
        """
        self.names = names
        self.compiled_regexes = compiled_regexes
        self.ratios = ratios
        # from IPython import embed;embed()

    def divide_into_param_groups(self, module: nn.Module):
        """divide into param_groups which need to be set to torch.optim.Optimizer
        
        Parameters
        ----------
        module : nn.Module
            module whose parameters are to be divided
        """
        compiled_regexes = self.compiled_regexes
        param_groups = divide_into_param_groups(module, compiled_regexes)

        return param_groups
    
    def multiply_lr(self, optimizer: optim.Optimizer):
        """Multiply lr 
        
        Parameters
        ----------
        optimizer : optim.Optimizer
            
        """
        lr_ratios = self.ratios
        multiply_lr(optimizer, lr_ratios)
        return optimizer


def divide_into_param_groups(module, compiled_regexes):
    # param_groups = [{'params': group_filter(module)} for group_filter in compiled_regexes]
    param_groups = [dict(params=list(),) for _ in range(len(compiled_regexes))]
    for ith, compiled_regex in enumerate(compiled_regexes):
        for param_name, param in module.named_parameters():
            if (compiled_regex.search(param_name) is not None):
                param_groups[ith]['params'].append(param)
    # from IPython import embed;embed()
            
    return param_groups

def multiply_lr(optimizer, lr_ratios, verbose=False):
    """ apply learning rate ratio for per-layer adjustment """
    assert len(optimizer.param_groups) == len(lr_ratios)
    for ith, (param_group, lr_ratio) in enumerate(zip(optimizer.param_groups, lr_ratios)):
        param_group['lr'] *= lr_ratio
        if verbose:
            print("%d params in param_group %d multiplied by ratio %.2g"%(len(param_group['params']), ith, lr_ratio))
    return optimizer

