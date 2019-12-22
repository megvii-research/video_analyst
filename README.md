# Video Analyst
This is the implementation of a series of basic algorithms which is useful for video understanding, including Single Object Tracking (SOT)
, Video Object Segmentation (VOS), etc.

Currnetly implemenation list:
* SiamFC++

## Quick start
### Test
```
python3 ./main/test.py --config 'experiments/siamfc++/siamfcpp_googlenet.yaml' --dataset 'VOT2018'
```

## Repository structure
```
├── experiments  # experiment configurations, in yaml format
├── main
│   ├── train.py # trainng entry point
│   └── test.py # test entry point
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

