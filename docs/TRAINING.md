# VideoAnalyst

## Training

By running [tools/train_test-alexnet.sh](../tools/train_test-alexnet.sh) or [tools/train_test-googlent.sh](../tools/train_test-googlenet.sh), a training process succeeded by a benchmarking process will be lauched.

Usage of Python script:

```Bash
python3 ./main/train.py --config 'path/to/config.yaml'
python3 ./main/test.py --config 'path/to/config.yaml'
```

Resuming from epoch number

```Bash
python3 ./main/train.py --config 'experiments/siamfcpp/train/siamfcpp_alexnet-trn.yaml' --resume-from-epoch=10
```

Resuming from snapshot file

```Bash
python3 ./main/train.py --config 'experiments/siamfcpp/train/siamfcpp_alexnet-trn.yaml' --resume-from-file='snapshots/siamfcpp_alexnet/epoch-10.pkl'
```

Training with PyTorch Distributed Data Parallel (DDP)

```Bash
python3 ./main/dist_train.py --config 'experiments/siamfcpp/train/siamfcpp_alexnet-dist_trn.yaml'
```

Configuration .yaml files are givin under [experiments/train/](../experiments/train/).

Before the training starts, the merged configuration file will be backed up at _EXP_SAVE/EXP_NAME/logs_.

### Training details

Harware configuration:

* #CPU: 32
* #GPU: 4
* Memory: 64 GiB

Several indexes related to training process have been listed in the table bellow:

|Experiment|Batch size|#Workers|#CPUs|#GPUs|Epoch speed|Iteration speed|
|---|---|---|---|---|---|---|
|siamfcpp-alexnet| 32 | 32| 32 | 4 |45min/epoch for epoch 0-19| 5it/s for /epoch 0-19 |
|siamfcpp-googlenet| 128 | 64 | 32 | 4 |20min/epoch for epoch 0-9; 24min/epoch for epoch 10-19 | 1.01it/s for epoch 0-9; 1.25s/it for epoch 10-19|
|siamfcpp-shufflenetv2x1_0| 32 | 32 | 32 | 4 |40min/epoch for epoch 0-19| 5it/s for epoch 0-19 |

## Stability

Stability test has been conducted on GOT-10k benchmark for our experiments (alexnet/googlenet/shufflenetv2x0.5/shufflenetv2x1.0). Concretely, for each experiment, we train on four different (virtual) PC and perform benchmarking on _val_ and _test_ subsets.

Results are listed as follows and they shall serve as reference for reproduction of the experiments by users of this code base.

### alexnet

| Site ID | Exp | Pipeline | Dataset | AO | Hardware |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | alexnet | SiamFCppTracker | GOT-10k-val | 72.1 | 2080ti |
| 2 | alexnet | SiamFCppTracker | GOT-10k-val | 72.4 | 2080ti |
| 3 | alexnet | SiamFCppTracker | GOT-10k-val | 71.9 | 2080ti |
| 4 | alexnet | SiamFCppTracker | GOT-10k-val | 72.2 | 1080ti |
| 1 | alexnet | SiamFCppTracker | GOT-10k-test | 53.8 | 2080ti |
| 2 | alexnet | SiamFCppTracker | GOT-10k-test | 53.6 | 2080ti |
| 3 | alexnet | SiamFCppTracker | GOT-10k-test | 51.2 | 2080ti |
| 4 | alexnet | SiamFCppTracker | GOT-10k-test | 53.7 | 1080ti |

### googlent

| Site ID | Exp | Pipeline | Dataset | AO | Hardware |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | googlenet | SiamFCppTracker | GOT-10k-val | 76.3 | 2080ti |
| 2 | googlenet | SiamFCppTracker | GOT-10k-val | 76.7 | 2080ti |
| 3 | googlenet | SiamFCppTracker | GOT-10k-val | 76.2 | 2080ti |
| 4 | googlenet | SiamFCppTracker | GOT-10k-val | 76.0 | 1080ti |
| 1 | googlenet | SiamFCppTracker | GOT-10k-test | 60.3 | 2080ti |
| 2 | googlenet | SiamFCppTracker | GOT-10k-test | 59.8 | 2080ti |
| 3 | googlenet | SiamFCppTracker | GOT-10k-test | 59.2 | 2080ti | 18e94f567a82bd482f64b8059a8e82c464629eb5 |
| 4 | googlenet | SiamFCppTracker | GOT-10k-test | 60.7 | 1080ti | db966bc51f420c9133385cb8e8deb281e555ac82 |

### shufflenetv2x0_5

| Site ID | Exp | Pipeline | Dataset | AO | Hardware |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-val | 73.1 | 2080ti |
| 2 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-val | 73.8 | 2080ti |
| 3 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-val | 73.0 | 2080ti |
| 4 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-val | 72.6 | 1080ti |
| 1 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-test | 53.5 | 2080ti |
| 2 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-test | 52.7 | 2080ti |
| 3 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-test | 53.2 | 2080ti |
| 4 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-test | 53.1 | 1080ti |

### shufflenetv2x1_0

| Site ID | Exp | Pipeline | Dataset | AO | Hardware |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-val | 76.3 | 2080ti |
| 2 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-val | 76.0 | 2080ti |
| 3 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-val | 76.1 | 2080ti |
| 4 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-val | 76.1 | 1080ti |
| 1 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-test | 57.2 | 2080ti |
| 2 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-test | 54.2 | 2080ti |
| 3 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-test | 55.4 | 2080ti |
| 4 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-test | 55.4 | 1080ti |
