# Video Analyst
This is the implementation of a series of basic algorithms which is useful for video understanding, including Single Object Tracking (SOT)
, Video Object Segmentation (VOS), etc.

Currnetly implemenation list:
* SiamFC++ (SOT)


## Setup
### Compile evaluation toolkit
```
cd videoanalyst/evaluation/vot_benchmark
bash make.sh  # compile to get test_utils/pyvotkit/region_XXX.so
cd -
```

### Set datasets
Set soft link to dataset directory
```
ln -s path_to_datasets datasets
```
At _path_to_datasets_:
```
path_to_datasets
└── VOT  # experiment configurations, in yaml format
    ├── vot2018
    │    ├── VOT2018
    │    │    ├── ...
    │    │    └── list.txt
    │    └── VOT2018.json
    └── vot2019
         ├── VOT2019
         │    ├── ...
         │    └── list.txt
         └── VOT2019.json
```
Auxilary files (list.txt / VOTXXXX.json) located at _videoanalyst/evaluation/vot_benchmark/vot_list_

### Set models
Set soft link to model directory
```
ln -s path_to_models models
```
At _path_to_models_
```
path_to_datasets
└── siamfc
    ├── alexnet
    │    └── epoch-19.pkl
    └── googlenet
         └── epoch-15.pkl
```

## Quick start
### Test
```
python3 ./main/test.py --config 'experiments/siamfc++/siamfcpp_googlenet.yaml' --dataset 'VOT2018'
```

## Repository structure
```
├── experiments  # experiment configurations, in yaml format
├── main
│   ├── train.py  # trainng entry point
│   └── test.py  # test entry point
├── video_analyst
│   ├── data  # modules related to data
│   │   ├── dataset  # data fetcher of each individual dataset
│   │   ├── sampler  # data sampler, including inner-dataset and intra-dataset sampling procedure
│   │   ├── dataloader.py  # data loading procedure
│   │   └── transformer  # data augmentation
│   ├── engine  # procedure controller, including traiing control / hp&model loading
│   │   ├── hook  # hook for tasks during training, including visualization / logging / benchmarking
│   │   ├── trainer.py  # train a epoch
│   │   ├── tester.py  # test a model on a benchmark
│   ├── model # model builder
│   │   ├── backbone  # backbone network builder
│   │   ├── task_model  # holistic model builder
│   │   ├── task_head  # head network builder
│   │   └── loss  # loss builder
│   ├── pipeline  # pipeline builder (tracking / vos)
│   ├── config  # configuration manager
│   ├── evaluation  # benchmark
│   ├── optimize # optimization-related module (learning rate, gradient clipping, etc.)
│   │   ├── lr_schedule # learning rate scheduler
│   │   ├── optimizer # optimizer
│   │   └── grad_modifier # gradient-related operation (parameter freezing)
│   └── utils  # useful tools
└── README.md
```

## Acknowledgement
* video_analyst/evaluation/vot_benchmark and other related code have been borrowed from [PySOT](https://github.com/STVIR/pysot)
* video_analyst/evaluation/vot_benchmark and other related code have been borrowed from [got-toolkit](https://github.com/got-10k/toolkit.git)
