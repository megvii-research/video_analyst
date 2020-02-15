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

Please refer to [docs/TEMPLATES/README.md](TEMPLATES/README.md) for detail.

## Configuration Tree

Based on [yaml](https://yaml.org/) and [yacs](https://github.com/rbgirshick/yacs), _videoanalyst_ arranges its configuration in a hierarchical way.

### Hyper-parameters

Developpers are recommended to take default _.yaml_ configuration files as example and start from them. Additionally, the code definitions as well as their descriptions are under _XXX_impl_ of each module.

Please refer to [docs/TEMPLATES/README.md](TEMPLATES/README.md) for detail.

## Add Your Own Module

Please refer to [docs/TEMPLATES/README.md](TEMPLATES/README.md) for detail.

Additioanlly, users are recommended to contribute their code under [contrib](../contrib/__init__.py). It is a good practice to seperate stable part and experimental part of the project.

## Structure

### Trainer

#### Trainer Structure

```File Tree
Trainer
├── Dataloder (pytorch)               # make batch of training data
│   └── AdaptorDataset (pytorch)      # adaptor class (pytorch index dataset)
│       └── Datapipeline              # integrate data sampling, data augmentation, and target making process
│           ├── Sampler               # define sampling strategy
│           │   ├── Dataset           # dataset interface
│           │   └── Filter            # define rules to filter out invalid sample
│           ├── Transformer           # data augmentation
│           └── Target                # target making
├── Optimizer
│   ├── Optimizer (pytorch)           # pytorch optimizer
│   │   ├── lr_scheduler              # learning rate scheduling
│   │   └── lr_multiplier             # learning rate multiplication ratio
│   ├── Grad_modifier                 # grad clip, dynamic freezing, etc.
│   └── TaskModel                     # model, subclass of pytorch module (torch.nn.Module)
│       ├── Backbone                  # feature extractor
│       ├── Neck                      # mid-level feature map operation (e.g. cross-correlation)
│       └── Head                      # task head (bbox regressor, mask decoder, etc.)│
└── Monitor                           # monitoring (e.g. pbar.set_description, tensorboard, etc.)
```

#### Trainer Building Process (Functional Representation)

```Python
model = builder.build(model_cfg)  # model_cfg.loss_cfg, model.loss
optimzier = builder.build(optim_cfg, model)
dataloader = builder.build(data_cfg)
trainer = builder.build(trainer_cfg, optimzier, dataloader)
```

### Tester

#### Tester Structure

```File Tree
Tester
├── Benchmark implementation          # depend on concrete benchmarks (e.g. VOT / GOT-10k / LaSOT / etc.)
└── Pipeline                          # manipulate underlying nn model and perform pre/post-processing
    └── TaskModel                     # underlying nereural network
        ├── Backbone                  # feature extractor
        ├── Neck                      # mid-level feature map operation (e.g. cross-correlation)
        └── Head                      # task head (bbox regressor, mask decoder, etc.)
```

Remarks:

- Pipeline object can run standalone. See [docs/PIPELINE_API.md](PIPELINE_API.md) for detail.

## Training

### Regex

### Scheduler

### Grad modifier

## Misc

### Logging

Names of currently used loggers are listed as below:

- global
  - built on each place
- data
  - built at: videoanalyst/data/builder.py
  - log file by default stored at: snapshots/EXP_NAME/logs/data.log
