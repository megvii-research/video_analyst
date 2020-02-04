# -*- coding: utf-8 -*
from typing import Dict

from yacs.config import CfgNode

def multiply_lr(optimizer, lr_ratios, verbose=False):
    """ apply learning rate ratio for per-layer adjustment """
    assert len(optimizer.param_groups) == len(lr_ratios)
    for ith, (param_group, lr_ratio) in enumerate(zip(optimizer.param_groups, lr_ratios)):
        param_group['lr'] *= lr_ratio
        if verbose:
            print("%d params in param_group %d multiplied by ratio %.2g"%(len(param_group['params']), ith, lr_ratio))

# def divide_into_param_groups(module, group_filters):
#     param_groups = [{'params': group_filter(module)} for group_filter in group_filters]
#     return param_groups

# def divide_into_param_groups(module, lr_multipliers):
#     filters = [lambda s: re.compile(lr_multiplier["regex"]).search(s) is not None
#                for lr_multiplier in lr_multipliers]
#     ratios = [lr_multiplier["ratio"] for lr_multiplier in lr_multipliers]
#     params = [[] for _ in range(len(filters+1))]
#     for name, param in module.named_parameters():
#         for ith, filt in enumerate(filters):
#             if filt(name):
#                 params[ith].append(param)
#             break
        

#     param_groups = [{'params': group_filter(module)} for group_filter in group_filters]
#     return param_groups

def resolve_lr_multiplier_cfg(cfg: CfgNode) -> Dict:
    schedule = dict()
    schedule["filter"] = lambda s: re.compile(cfg.regex).search(s) is not None
    schedule["ratio"] = cfg.ratio
    return schedule
