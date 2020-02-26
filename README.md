# Video Analyst

[![Build Status](https://travis-ci.org/MegviiDetection/video_analyst.svg?branch=master)](https://travis-ci.org/MegviiDetection/video_analyst)

This is the implementation of a series of basic algorithms which is useful for video understanding, including Single Object Tracking (SOT), Video Object Segmentation (VOS), etc.

Current implementation list:

* SOT
  * [SiamFC++: Towards Robust and Accurate Visual Tracking with Target Estimation Guidelines](https://arxiv.org/abs/1911.06188) [[demo]](https://www.youtube.com/watch?v=TCziWahnXT8&list=PL4KqNq8e6fJkfk35zHRaUd21ExV522JK0&index=4&t=0s&app=desktop)
<div align="center">
  <img src="docs/resources/siamfcpp_ice2.gif" width="800px" />
  <p>Example SiamFC++ outputs.</p>
</div> 

## Quick start

### Setup

Please refer to [SETUP.md](docs/SETUP.md)

### Test

#### Test on VOT

```Bash
python3 ./main/test.py --config 'experiments/siamfcpp/test/vot/siamfcpp_alexnet.yaml'
```

#### Test on GOT-10k

```Bash
python3 ./main/test.py --config 'experiments/siamfcpp/test/got10k/siamfcpp_alexnet-got.yaml'
```

Please refer to [docs/TEST.md](docs/TEST.md) for detail.

### Training

#### From epoch 0

```Bash
python3 ./main/train.py --config 'experiments/siamfcpp/train/siamfcpp_alexnet-trn.yaml'
```

#### Resuming (e.g. from epoch 10)

```Bash
python3 ./main/train.py --config 'experiments/siamfcpp/train/siamfcpp_alexnet-trn.yaml' --resume-from-epoch=10
```

Please refer to [docs/TRAINING.md](docs/TRAINING.md) for detail.

## Repository structure (in progress)

```File Tree
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
│   │   ├── monitor  # monitor for tasks during training, including visualization / logging / benchmarking
│   │   ├── trainer.py  # train a epoch
│   │   ├── tester.py  # test a model on a benchmark
│   ├── model # model builder
│   │   ├── backbone  # backbone network builder
│   │   ├── common_opr  # shared operator (e.g. cross-correlation)
│   │   ├── task_model  # holistic model builder
│   │   ├── task_head  # head network builder
│   │   └── loss  # loss builder
│   ├── pipeline  # pipeline builder (tracking / vos)
│   │   ├── segmenter  # segmenter builder for vos
│   │   ├── tracker  # tracker builder for tracking
│   │   └── utils  # pipeline utils
│   ├── config  # configuration manager
│   ├── evaluation  # benchmark
│   ├── optim  # optimization-related module (learning rate, gradient clipping, etc.)
│   │   ├── optimizer # optimizer
│   │   ├── scheduler # learning rate scheduler
│   │   └── grad_modifier # gradient-related operation (parameter freezing)
│   └── utils  # useful tools
└── README.md
```

## docs

For detail, please refer to markdown files under _docs_.

* [SETUP.md](docs/SETUP.md): instructions for setting-up
* [MODEL_ZOO.md](docs/MODEL_ZOO.md): description of released models
* [TRAINING.md](docs/TRAINING.md): details related to training
* [DEVELOP.md](docs/DEVELOP.md): description of project design (registry, configuration tree, etc.)
* [PIPELINE_API.md](docs/PIPELINE_API.md): description for pipeline API
* [FORMATTING_INSTRUCTION](docs/FORMATTING_INSTRUCTIONS.md): instruction for code formatting (yapf/isort/flake/etc.)

## TODO

* [] Training code
  * [] LaSOT
  * [] COCO
* [] Test code for OTB, LaSOT, TrackingNet

## Acknowledgement

* video_analyst/evaluation/vot_benchmark and other related code have been borrowed from [PySOT](https://github.com/STVIR/pysot)
* video_analyst/evaluation/got_benchmark and other related code have been borrowed from [got-toolkit](https://github.com/got-10k/toolkit.git)

## Contact

Maintainer (sorted by family name):

* Xi Chen[@XavierCHEN34](https://github.com/XavierCHEN34)
* Zuoxin Li[@lzx1413](https://github.com/lzx1413)
* Zeyu Wang[@JWarlock](http://github.com/JWarlock)
* Yinda Xu[@MARMOTatZJU](https://github.com/MARMOTatZJU)

