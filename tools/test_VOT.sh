#!/usr/bin/env bash

python3 ./main/test.py --config 'experiments/siamfcpp/test/siamfcpp_alexnet.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/test/siamfcpp_alexnet-multi_temp.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/test/siamfcpp_googlenet.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/test/siamfcpp_googlenet-multi_temp.yaml'
