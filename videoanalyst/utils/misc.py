from typing import Dict
import logging
import os
import time

from yacs.config import CfgNode as CN


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict, print(
        module_name, module_dict, 'defined in several script files')
    module_dict[module_name] = module


class Registry(dict):
    r"""
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    usually declared in XXX_base.py, e.g. videoanalyst/model/backbone/backbone_base.py

    used as decorator when declaring the module:

    @some_registry.register
    def foo():
        ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
    """
    TAG = 'global'
    logger = logging.getLogger(TAG)

    def __init__(self, *args, **kwargs):
        self.name = 'Registry'
        if len(args) > 0 and isinstance(args[0], str):
            name, *args = args
            self.name = name
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module):
        name = module.__name__
        _register_generic(self, name, module)
        self.logger.info('%s: %s registered' % (self.name, name))
        return module


def load_cfg(path: str):
    r"""
    Load yaml with yacs

    Arguments
    ---------
    path: str
        yaml path
    """
    with open(path, 'r') as f:
        config_node = CN.load_cfg(f)

    return config_node

def merge_cfg_into_hps(cfg: CN, hps: Dict):
    for hp_name in hps:
        if hp_name in cfg:
            new_value = cfg[hp_name]
            hps[hp_name] = new_value
    return hps

class Timer():
    r"""
    Mesure & print elapsed time witin environment
    """
    def __init__(self, info='', enable=True):
        r"""
        Arguments
        ---------
        :param info: prompt to print(will be appended with "elapsed time: %f")
        :param enable: enable timer or not
        """
        self.info = info
        self.enable = enable

    def __enter__(self, ):
        if self.enable:
            self.tic = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            self.toc = time.time()
            print('%s elapsed time: %f'%(self.info, self.toc-self.tic))
