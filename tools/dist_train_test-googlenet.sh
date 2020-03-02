#!/usr/bin/env bash
python3 -W ignore ./main/dist_train.py --config 'experiments/siamfcpp/train/siamfcpp_googlenet-dist_trn.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/train/siamfcpp_googlenet-trn.yaml'
