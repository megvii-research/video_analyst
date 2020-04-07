#!/usr/bin/env bash
python3 ./main/dist_train.py --config 'experiments/siamfcpp/train/fulldata/siamfcpp_googlenet-dist_trn-fulldata.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/train/fulldata/siamfcpp_googlenet-dist_trn-fulldata.yaml'
