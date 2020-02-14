# PIPELINE_API

## Minimal runnable pipeline

Suppose that:

* you have your .yaml file in the string _exp_cfg_path_ and videoanalyst at the same level;
* you have a GPU with index 0,
then the following code segment will instantiate and configure a pipeline object which can be used immediately for your own application. It supports following APIs:
* _void init(im, state)_
* _state update(im)_

```Python
import torch

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder

root_cfg.merge_from_file(exp_cfg_path)

# resolve config
task, task_cfg = specify_task(root_cfg)
task_cfg.freeze()

# build model
model = model_builder.build_model(task, task_cfg.model)
# build pipeline
pipeline = pipeline_builder.build_pipeline('track', task_cfg.pipeline, model)
pipeline.to_device(torch.device("cuda:0"))
```
