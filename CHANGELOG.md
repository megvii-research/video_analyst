# Changelog

Date format: YY/MM/DD

## 2020/02/15

* Add contrib
  * Directory under root & template

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
