# -*- coding: utf-8 -*
import logging
from typing import List, Dict

from yacs.config import CfgNode

from .process_base import TASK_PROCESSES, ProcessBase
from videoanalyst.utils.misc import merge_cfg_into_hps

def build(task: str, cfg: CfgNode) -> List[ProcessBase]:
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        node name: processes
    
    Returns
    -------
    List[ProcessBase]
        list of processes
    """
    assert task in TASK_PROCESSES, "no tester for task {}".format(task)
    modules = TASK_PROCESSES[task]

    names = cfg.names
    processes = []
    for name in names:
        process = modules[name]()
        hps = process.get_hps()
        hps = merge_cfg_into_hps(cfg[name], hps)
        process.set_hps(hps)
        process.update_params()
        processes.append(process)

    return processes

def get_config() -> Dict[str, CfgNode]:
    cfg_dict = {name: CfgNode() for name in TASK_PROCESSES.keys()}

    for cfg_name, modules in TASK_PROCESSES.items():
        cfg = cfg_dict[cfg_name]
        cfg["names"] = [""]

        for name in modules:
            cfg[name] = CfgNode()
            module = modules[name]
            hps = module.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]

    return cfg_dict
