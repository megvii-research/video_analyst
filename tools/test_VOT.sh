#!/usr/bin/env bash

python3 ./main/test.py --config 'experiments/siamfcpp/siamfcpp_alexnet.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/siamfcpp_alexnet-multi_temp.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/siamfcpp_googlenet.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/siamfcpp_googlenet-multi_temp.yaml'
