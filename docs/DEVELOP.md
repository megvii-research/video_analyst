# DEVELOP

## Registry Mechanism

It is a common design that is adopted by several mainsteam deep learning repositries such as [MMDetection](https://github.com/open-mmlab/mmdetection) and [Detectron2](https://github.com/facebookresearch/detectron2).

The main idea is to build a dictionary whose _keys_ are module names and whose _values_ are the module class objects, and then construct the whole pipeline (e.g. tracker/segmenter/trainer of them/etc.) by retrieving module class objects and instantiating them with predefiend configurations.

An example for demonstrating the usage the registry is given as below:

```Python
# In XXX_base.py
from videoanalyst.utils import Registry

TRACK_TEMPLATE_MODULES = Registry('TRACK_TEMPLATE_MODULE')
VOS_TEMPLATE_MODULES = Registry('VOS_TEMPLATE_MODULE')

TASK_TEMPLATE_MODULES = dict(
    track=TRACK_TEMPLATE_MODULES,
    vos=VOS_TEMPLATE_MODULES,
)

# In XXX_impl/YYY.py
@TRACK_TEMPLATE_MODULES.register
@VOS_TEMPLATE_MODULES.register
class TemplateModuleImplementation(TemplateModuleBase):
...
```

## Configuration Tree
Based on [yaml](https://yaml.org/) and [yacs](https://github.com/rbgirshick/yacs), _videoanalyst_ arranges its configuration in a hierarchical way.
