# SOT TEST 

## Usage of Test.py

Change directory to the repository root.

```Bash
python main/test.py --config experiments/siamfcpp/test/got10k/siamfcpp_alexnet-got.yaml
```

## Test Scripts

A collection of test scripts are located under _tools/test/_:

- [tools/test/test_VOT.sh](../../tools/siamfcpp/test/test_VOT.sh)
- [tools/test/test_GOT.sh](../../tools/siamfcpp/test/test_GOT.sh)
- [tools/test/test_LaSOT.sh](../../tools/siamfcpp/test/test_LaSOT.sh)
- [tools/test/test_OTB.sh](../../tools/siamfcpp/test/test_OTB.sh)

## Check test results

_EXP_NAME_ is the string value of key _test.track.exp_name_ in the corresponding _.yaml_ file.

### Check VOT results

```Bash
view logs/VOT2018/<EXP_NAME>.csv
```

### Check GOT-Benchmark results

GOT-Benchmark contains testers for a series of benchmarks, including OTB, VOT, LaSOT, GOT-10k, TrackingNet.

```Bash
view logs/GOT-Benchmark/report/GOT-10k/<EX015P_NAME>/performance.json
view logs/GOT-Benchmark/report/LaSOT/<EXP_NAME>/performance.json
view logs/GOT-Benchmark/report/otb2015/<EXP_NAME>/performance.json
view logs/GOT-Benchmark/report/TrackingNet/<EXP_NAME>/performance.json
```
### Speed up with TensorRT

1. we adopt [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) accelerate the speed. Because of this [issue](https://github.com/NVIDIA-AI-IOT/torch2trt/issues/251)， we only transfer the feature extraction part yet.

2. please refer https://github.com/researchmm/TracKit/blob/master/lib/tutorial/install_trt.md to install torch2trt.
3. we provide trt model for siamfcpp-googlenet-vot now, you can set trt_mode True in [yaml](../../experiments/siamfcpp/test/vot/siamfcpp_googlenet.yaml) to enable it.
4. You can refer to [cvt_trt.py](../../main/cvt_trt.py) to transfer other models to trt.

## Misc

### Use multiple GPUs for test

Consider changing _device_num_ in .yaml configuration file.

```yaml
tester:
    names: ["GOT10kTester",]
    GOT10kTester:
    exp_name: *TEST_NAME
    exp_save: *TEST_SAVE
    device_num: 4  # change here to use four GPU
    subsets: ["val"]  # (val|test)
```
