import os
import sys

import videoanalyst.model.builder as model_builder
from videoanalyst.config.config import cfg as whole_cfg
from videoanalyst.config.config import specify_task

root_dir = os.path.abspath("./")
sys.path.insert(0, root_dir)


whole_cfg.merge_from_file("experiments/siamfc++/alexnet_siamfc++.yaml")
print(whole_cfg)
task, config = specify_task(whole_cfg)
print(task, config)
model = model_builder.build_model(task, config.model)
print(model)
