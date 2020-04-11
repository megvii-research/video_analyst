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

### python-tkinter

In case of:

```Python
ModuleNotFoundError: No module named 'tkinter'
```

Please install python3-tk by running:

```Bash
sudo apt-get install python3-tk
```

## Download models & raw results

* [Google Drive](https://drive.google.com/open?id=1UXshq4k9WKx4hNkdpOagJLXPR57ZkBkg)
* [baidu yun](https://pan.baidu.com/s/1uZ26iZyVJm50dJ3GoLCQ9w), code: rcsn

## Compile evaluation toolkit

```Bash
bash compile.sh
```

## Set datasets
Download [Davis](https://davischallenge.org/davis2017/code.html)
Set soft link to dataset directory 

```bash
ln -s path_to_datasets datasets/DAVIS
```

At _path_to_datasets_:

```File Tree
datasets
└── DAVIS
    ├── Annotations
    ├── ImageSets
    └── ...
```


## Set models

Set soft link to model directory

```Bash
ln -s path_to_models models/sat
```

At _path_to_models_:

```File Tree
models
├── sat
    |── transferred-model
       ├── siamfcpp_brain2torch-md5_abfe30e8531e001d83e9f0a6482da444.pkl
       ├── siamfcpp-googlenet-vot-md5_f2680ba074213ee39d82fcb84533a1a6.pkl
```
