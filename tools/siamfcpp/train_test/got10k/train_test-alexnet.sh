#!/usr/bin/env bash
python3 ./main/train.py --config 'experiments/siamfcpp/train/got10k/siamfcpp_alexnet-trn.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/train/got10k/siamfcpp_alexnet-trn.yaml'
