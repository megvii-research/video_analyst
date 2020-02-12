# Setup

## Install requirements

- Linux or MacOS
- Python >= 3.5
- GCC >= 4.9

```Bash
git clone https://github.com/MegviiDetection/video_analyst.git
cd video_analyst
```

You can choose either using native python (with pip/pip3) or using virtual environment (with conda).

```Bash
pip3 install -U -r requirements.txt
```

## Compile evaluation toolkit

```Bash
bash compile.sh
```

## Set datasets

Set soft link to dataset directory (see [config example](../experiments/siamfcpp/test/vot/siamfcpp_alexnet.yaml))

```bash
ln -s path_to_datasets datasets
```

At _path_to_datasets_:

```File Tree
path_to_datasets
├── VOT  # experiment configurations, in yaml format
│   ├── vot2018
│   │    ├── VOT2018
│   │    │    ├── ...
│   │    │    └── list.txt
│   │    └── VOT2018.json
│   └── vot2019
│        ├── VOT2019
│        │    ├── ...
│        │    └── list.txt
│        └── VOT2019.json
└── GOT-10k  # same structure as full_data.zip
    ├── train
    ├── val
    └── test
```

### Auxilary files for VOT

Auxilary files (list.txt / VOTXXXX.json) located at _videoanalyst/evaluation/vot_benchmark/vot_list_

Script for copying:

```Bash
# Assuming that VOT benchmarks located under datasets/VOT and your
cd $(git rev-parse --show-toplevel)  # change directory to the repo. root
cp videoanalyst/evaluation/vot_benchmark/vot_list/vot2018/VOT2018.json datasets/VOT/vot2018/
cp videoanalyst/evaluation/vot_benchmark/vot_list/vot2018/list.json datasets/VOT/vot2018/VOT2018/
```

### Download Links for Datasets

We provide download links for VOT2018 / VOT2019:

- [Google Drive](https://drive.google.com/open?id=18vaGhvrr_rt70sZr_TisrWl7meO9NE0J)
- [Baidu Disk](https://pan.baidu.com/s/1HZkbWen4mEkxaJL3Rj9pig), code: xg4q

__Acknowledgement:__: Following datasets have been downloaded with [TrackDat](https://github.com/jvlmdr/trackdat)

- VOT2018
- VOT2019

## Set models

Set soft link to model directory

```Bash
ln -s path_to_models models
```

At _path_to_models_:

```File Tree
path_to_models
├── siamfcpp
│   ├── siamfcpp-alexnet-vot-md5_18fd31a2f94b0296c08fff9b0f9ad240.pkl
│   ├── siamfcpp-googlenet-vot-md5_f2680ba074213ee39d82fcb84533a1a6.pkl
│   ├── siamfcpp-tinyconv-vot-md5_cb9c2e8c7851ebf79677522269444cb2.pkl
│   ├── ...
│   ...
├── alexnet
│   └── alexnet-nopad-bn-md5_fa7cdefb48f41978cf35e8c4f1159cdc.pkl
├── googlenet
│   └── inception_v3_google-1a9a5a14-961cad7697695cca7d9ca4814b17a88d.pth
│
```
