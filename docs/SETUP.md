## Setup
### 
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

### Set VOT datasets
Set soft link to dataset directory (see [config example](../experiments/siamfc++/siamfcpp_alexnet.yaml))
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

At _path_to_models_:
```
path_to_datasets
└── siamfcpp
    ├── alexnet
    │    └── epoch-19.pkl
    └── googlenet
         └── epoch-15.pkl
```
