import os

def ensure(dir_path: str):
    if os.path.exists(dir_path):
        return
    else:
        os.makedirs(dir_path)

def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict, print(module_name, module_dict)
    module_dict[module_name] = module


class Registry(dict):
    """
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    used as decorator when declaring the module:

    @some_registry.register
    def foo():
        ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
    """
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module):
        _register_generic(self, module.__name__, module)
        return module

