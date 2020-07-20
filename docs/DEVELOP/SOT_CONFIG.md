# Configuration yaml


Example _experiments/siamfcpp/train/got10k/siamfcpp_alexnet-trn.yaml_

## Training

### How to modify the number of epochs

- Modify the value in YAML
  - train.track.data.num_epochs
  - L88: _num_epochs: &NUM_EPOCHS 20_
- Modify the value in JSON (lr_policy)
  - train.track.optim.optimizer.SGD.lr_policy
  - L172: _"max_epoch": 19_

### How to modify the nmber of image pairs of each epoch

- Modify the value in YAML
  - train.track.data.nr_image_per_epoch
  - L91: _nr_image_per_epoch: &NR_IMAGE_PER_EPOCH 400000_

### How to modify the learning rate

- Modify the value in JSON (lr_policy)
  - train.track.optim.optimizer.SGD.lr_policy
  - L164: _"end_lr": 0.08,_
  - L170: _"start_lr": 0.08,_

#### How to modify the learning rate in backbone

- Modify the value in JSON (lr_policy)
  - train.track.optim.optimizer.SGD.lr_multiplier
  - L179: _"ratio": 0.1_

### How to change the dynamic freezing schedule

- Modify the value in JSON (lr_policy)
  - train.track.optim.grad_modifier
