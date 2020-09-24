# Setup

Followings are steps that need to be taken before running any part of our codebase.

## Install requirements

- Linux or MacOS
- Python >= 3.5
- GCC >= 4.9

Fetch code from our repo

```Bash
git clone https://github.com/MegviiDetection/video_analyst.git
cd video_analyst
```

You can choose either using native python (with pip/pip3) or using virtual environment.

```Bash
python setup.py develop
pre-commit install
```


### GPU assignement

By default, code use all GPUs visible for Python program with index starting from 0.

Use _CUDA_VISIBLE_DEVICES_ to designate arbitrary GPUs to program via GPU visbility control.

For example, to assign GPU 2 & GPU 3 for training, launch training code with:

```bash
CUDA_VISIBLE_DEVICES=2,3 python train.py --config experiments/siamfcpp/test/got10k/siamfcpp_alexnet-got.yaml
```

c.f. https://stackoverflow.com/questions/39649102/how-do-i-select-which-gpu-to-run-a-job-on




## Troubleshootings

### python-tkinter

In case of:

```Bash
ModuleNotFoundError: No module named 'tkinter'
```

Please install python3-tk by running:

```Bash
sudo apt-get install python3-tk
```


### Matplotlib backend

Incase of:

```Bash
ImportError: Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'headless' is currently running
```

Please change Matplotlib's backend by setting:

```Bash
export MPLBACKEND=TkAgg
```

### pycocotools

The following code ensure the installation of cocoapi in any case.

pip3 install --user pycocotools -i https://pypi.tuna.tsinghua.edu.cn/simple

However, the pycocotools hosted on Pypi may be incompatible with numpy==1.18.0. This issue has been fixed in [this commit](https://github.com/cocodataset/cocoapi/commit/6c3b394c07aed33fd83784a8bf8798059a1e9ae4). If you have numpy==1.18.0 installed (instead numpy==1.16.0 in our requirements.txt), please install pycocotools from the [official repo](https://github.com/cocodataset/cocoapi) on Github.

```Bash
git clone https://github.com/cocodataset/cocoapi
cd cocoapi
make
make install  # may need sudo if it fails
```



## Auto mixed precision training

AMP is supported from pytorch >= 1.6. It can reduce about 50% of the GPU memory usage on RTX2018Ti without preformance drop. If you want to enable it, just set the amp `True` as in [config](../../experiments/siamfcpp/train/got10k/siamfcpp_alexnet-trn.yaml). `SigmoidCrossEntropyRetina` should be replaced by `FocalLoss` if you want to enable amp.
