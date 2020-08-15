# Changelog

Date format: YY/MM/DD

## 2020/08/15
* add atom online module from pytracking & DROL
* update speed defination in vot test

## 2020/08/08
* adopt pre-commit to format code while committing
* reformat code with isort, autoflake and yapf

## 2020/07/30
* support amp from pytorch>=1.6

## 2020/06/29
* support acclerating tracking by tensorRT

## 2020/06/23

* refine make_densebox_target.py
  * target making code: numpy -> pytorch
  * simplified for SOT task
    * reduce to single resolution layer
* relax restrictions in requirements.txt
  * e.g. version of pytorch, opencv, numpy, etc.

## 2020/05/08
* format evaluation code
* Long-term tracking 
   * add dataset and test_engine for UAV123 and VOTLT
   * add test results for UAV123
   * add long-term trakcker (need to  be refined)
   
## 2020/04/28

* Modify interface of _siamfcpp_track.py: update_
  * Accept provided target state (bbox)

## 2020/04/25

* Add multi-processing tester
  * otb
  * lasot
  * trackingnet
  * got10k
  
## 2020/04/18
* add SAT train code
* update SAT test code
* change TrackInfo to TextInfo
* new param loader for model borrowed from dectron2
* update coco dataset from detectron2

## 2020/04/18
* New features in SOT demo (sot_video.py)
  * Add image file video reader (reading from image files)
  * Add image file video dumping (dump result as frame-level image files)

## 2020/04/13
* Add OTB test

## 2020/04/12
* Add iterable dataset in
  * dataset_type under config node: data
    * regular: former dataset
    * iterable: newly added dataset
  * data/adaptor_dataset.py: _AdaptorIterableDataset_ / _MultiStreamDataloader_
  * data/builder.py: the branch with _cfg.dataset_type_
* Fix seed issue
  * pass builder's seed to dataset as ext_seed

## 2020/04/06
* Add model/pipeline/tester for SAT on DAVIS
* Add transferred models for SAT

## 2020/03/25

____Major changes____

* Add hyper-parameter search script
  * _main/hpo.py_
  * _experiment_
  * _videoanalyst/utils/hpo.py_

## 2020/03/23

____Major changes____

* Add full training data
  * coco / det / vid / trackingnet / lasot / got10k
  * with caching mechanism (reducing storage I/O by caching indexes into memory)
  * TODO: perform verification with full data (under batchsize=256)

## 2020/03/25

____Major changes____

* Add hyper-parameter search script
  * _main/hpo.py_
  * _experiment_
  * _videoanalyst/utils/hpo.py_

## 2020/03/23

____Major changes____

* Add full training data
  * coco / det / vid / trackingnet / lasot / got10k
  * with caching mechanism (reducing storage I/O by caching indexes into memory)
  * TODO: perform verification with full data (under batchsize=256)

## 2020/03/18

____Major changes____

* Add caching in LaSOT dataset
* Add multi-mode in siamfcpp visualization debugging tools
  * demo/main/debug/visualize_siamfcpp_training_data.py
  * Seperate config file for debugging: experiments/siamfcpp/data/siamfcpp_data-trn.yaml

## 2020/03/17
  
____Major changes____

* Update all training configuration
  * the cuda device number is defined by num_processes rather than devices in training phase
  * the settings between DP and DDP are unified now
* Adopt loguru to replace logging for better logger
* Update code for distributed training
* Update code for model/pipeline/trainer

## 2020/03/14

____Major changes____

* Update all training configuration
  * learning rate multiplied by 2 in order to be compatible with PyTorch==1.4.0 & CUDA==10.1
  * Update requirements.txt
* Add tinyconv training configuration
  * _experiments/siamfcpp/train/siamfcpp_tinyconv-trn.yaml_
* Add dataloader visualization tools for SiamFC++ (SOT)
  * _demo/main/debug/visualize_siamfcpp_training_data.py_

## 2020/03/07

____Major changes____

* Add webcam demo
  * _demo/main/video/sot_video.py_

## 2020/03/02

____Major changes____

* Add training with PyTorch Distributed Data Parallel (DDP)
  * _main/dist_trian.py_
  * _.../trainer_impl/distributed_regular_trainer.py_

## 2020/02/27

____Major changes____

* Add TensorboardLogger in __monitor__
  * Recursively put engine_data as scalar into Tensorboard

## 2020/02/22

____Major changes____

* Caching mechanism added to [got10k dataset](videoanalyst/evaluation/got_benchmark/datasets/got10k.py)
  * Build cache at _root_dir/subset.pkl_ (default), subset=[train, val, test]
  * Cache created/loaded in GOT10k.data_dict and will be queried every time _\_\_getitem\_\__ is called
* Remove _num_iterations_ item in training .yaml files
  * replaced by value calculated with _nr_image_per_epoch_ and _minibatch_
    * _num_iterations_ = _nr_image_per_epoch_ // _minibatch_
  * all training configs has been updated.

____Minor changes____

* pipeline.builder's builder method has been changed
  * _build_pipeline_ -> _build_

## 2020/02/15

* Add contrib module's template
  * _docs/TEMPLATES/contrib_module_

## 2020/02/14

* Add one-shot detection demo
  * API (see [docs/PIPELINE_API.md](docs/PIPELINE_API.md)
  * Runnable demo
* Complete ShuffleNetV2x1.0 experiment

## 2020/02/13

* Support Training and Test on GOT-10k for Single Object Tracking
  * SiamFC++-AlexNet
  * SiamFC++-GoogLeNet
  * SiamFC++-ShuffleNetV2
* Training Components
  * Dataset helper, data sampler, transformer (data augmentation), etc.
  * Optimization: learing rate scheduler, dynamic freezing, etc.
  * Trainer & Monitor

## 2019/12/31

* Support SiamFC++ test on VOT benchmark
  * SiamFC++-AlexNet
  * SiamFC++-GoogLeNet

