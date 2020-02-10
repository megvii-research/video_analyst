# VideoAnalyst

## Training

Please refer to (tools/train-alexnet.sh) and (tools/train-googlent.sh) for command in detail.

Configuration .yaml files are givin under [this directory](experiments/train/)

### Training details

Configuration:

* #CPU: 32
* #GPU: 4
* Memory: 80 GiB

Several indexes related to training process have been listed in the table bellow:

|Experiment|Batch size|#Workers|#CPUs|#GPUs|Epoch speed|Iteration speed|
|---|---|---|---|---|---|---|
|siamfcpp-googlenet| 32 | 32| 32 | 4 |45min/epoch for epoch 0-19| 5it/s for /epoch 0-19 |
|siamfcpp-googlenet| 64 | 64 | 32 | 4 |40min/epoch for epoch 0-19| 4it/s for epoch 0-19 |
|siamfcpp-googlenet| 128 | 64 | 32 | 4 |20min/epoch for epoch 0-9; 24min/epoch for epoch 10-19 | 1.01it/s for epoch 0-9; 1.25s/it for epoch 10-19|
