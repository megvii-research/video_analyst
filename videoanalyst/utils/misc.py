import logging
import os

from yacs.config import CfgNode as CN


def ensure_dir(dir_path: str):
    """
    Ensure the existence of path (i.e. mkdir -p)
    :param dir_path: path to be ensured
    :return:
    """
    if os.path.exists(dir_path):
        return
    else:
        os.makedirs(dir_path)


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict, print(
        module_name, module_dict, 'defined in several script files')
    module_dict[module_name] = module


class Registry(dict):
    """
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
    """
    Load yaml with yacs
    :param path: yaml path
    :return:
    """
    with open(path, 'r') as f:
        config_node = CN.load_cfg(f)

    return config_node
