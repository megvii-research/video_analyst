## Setup
### Install requirements
You can choose either using native python (with pip/pip3) or using virtual environment (with conda). The command using pip3 and THU TUNA source is presented bellow:
```
pip3 install -U -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Compile evaluation toolkit
```
cd video_analyst/evaluation/vot_benchmark
bash make.sh  # compile to get test_utils/pyvotkit/region_XXX.so
cd -
```

### Set datasets
Set soft link to dataset directory (see _video_analyst/config.yaml_)
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
Auxilary files (list.txt / VOTXXXX.json) located at _video_analyst/evaluation/vot_benchmark/vot_list_

### Set models
Set soft link to model directory
```
ln -s path_to_models models
```

At _path_to_models_:
```
path_to_datasets
└── siamfc
    ├── alexnet
    │    └── epoch-19.pkl
    └── googlenet
         └── epoch-15.pkl
```
