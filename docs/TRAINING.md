# VideoAnalyst

## Training

By running [tools/train_test-alexnet.sh](../tools/train_test-alexnet.sh) or [tools/train_test-googlent.sh](../tools/train_test-googlenet.sh), a training process succeeded by a benchmarking process will be lauched.

Usage of Python script:

```Bash
python3 ./main/train.py --config 'path/to/config.yaml'
python3 ./main/test.py --config 'path/to/config.yaml'
```

Configuration .yaml files are givin under [experiments/train/](../experiments/train/)

### Training details

Harware configuration:

* #CPU: 32
* #GPU: 4
* Memory: 64 GiB

Several indexes related to training process have been listed in the table bellow:

|Experiment|Batch size|#Workers|#CPUs|#GPUs|Epoch speed|Iteration speed|
|---|---|---|---|---|---|---|
|siamfcpp-alexnet| 32 | 32| 32 | 4 |45min/epoch for epoch 0-19| 5it/s for /epoch 0-19 |
|siamfcpp-alexnet| 64 | 64 | 32 | 4 |40min/epoch for epoch 0-19| 4it/s for epoch 0-19 |
|siamfcpp-googlenet| 128 | 64 | 32 | 4 |20min/epoch for epoch 0-9; 24min/epoch for epoch 10-19 | 1.01it/s for epoch 0-9; 1.25s/it for epoch 10-19|
