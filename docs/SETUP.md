## Setup
### Install requirements
- Linux or MacOS
- Python >= 3.5
- GCC >= 4.9
```
git clone https://github.com/MegviiDetection/video_analyst.git
cd video_analyst
```
You can choose either using native python (with pip/pip3) or using virtual environment (with conda).
```
pip3 install -U -r requirements.txt
```

### Compile evaluation toolkit
```
bash compile.sh
```

### Set datasets
Set soft link to dataset directory (see [config example](../experiments/siamfcpp/siamfcpp_alexnet.yaml))
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

#### Download
We provide download links for VOT2018 / VOT2019:
* [Google Drive](https://drive.google.com/open?id=18vaGhvrr_rt70sZr_TisrWl7meO9NE0J)
* [Baidu Disk](https://pan.baidu.com/s/1HZkbWen4mEkxaJL3Rj9pig), code: xg4q

__Acknowledgement:__: Following datasets have been downloaded with [TrackDat](https://github.com/jvlmdr/trackdat) 
* VOT2018
* VOT2019

### Set models
Set soft link to model directory
```
ln -s path_to_models models
```

At _path_to_models_:
```
path_to_datasets
└── siamfcpp
    ├── alexnet
    │    └── epoch-19.pkl
    └── googlenet
         └── epoch-15.pkl
```
