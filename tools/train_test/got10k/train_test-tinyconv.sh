#!/usr/bin/env bash
python3 ./main/train.py --config 'experiments/siamfcpp/train/siamfcpp_tinyconv-trn.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/train/siamfcpp_tinyconv-trn.yaml'
