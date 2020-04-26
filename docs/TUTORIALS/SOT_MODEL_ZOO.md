# MODEL_ZOO

## Download links

Models & Raw results:

* [Google Drive](https://drive.google.com/open?id=1XhWIU1KIt9wvFpzZqEDaX-GrgZ9AVcOC)
* [Tencent Weiyun](https://share.weiyun.com/56C92l4), code: wg47g7

## Models

### VOT2018

VOT test configuration directory: _experiments/siamfcpp/test/vot_

| Backbone | Pipeline | Dataset | A | R | EAO | FPS@GTX2080Ti | FPS@GTX1080Ti | Config. Filename | Model Filename |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AlexNet | SiamFCppTracker | VOT2018 |0.588 | 0.243 | 0.373| ~200| ~185 | siamfcpp_alexnet.yaml | siamfcpp-alexnet-vot-md5_18fd31a2f94b0296c08fff9b0f9ad240.pkl|
| AlexNet | SiamFCppTracker | VOT2018 |0.576 | 0.183 | 0.393| ~200| ~185 | siamfcpp_alexnet-new.yaml | siamfcpp-alexnet-vot-md5_88e4e9ee476545b952b04ae80c480f08.pkl|
| AlexNet | SiamFCppMultiTempTracker| VOT2018 | 0.597 | 0.215 | 0.370 | ~90 | ~75 | siamfcpp_alexnet-multi_temp.yaml | siamfcpp-alexnet-vot-md5_18fd31a2f94b0296c08fff9b0f9ad240.pkl|
| GoogLeNet | SiamFCppTracker | VOT2018 | 0.583 | 0.173 | 0.426 | ~80 | ~65 | siamfcpp_googlenet.yaml | siamfcpp-googlenet-vot-md5_f2680ba074213ee39d82fcb84533a1a6.pkl |
| GoogLeNet | SiamFCppTracker | VOT2018 | 0.588 | 0.183 | 0.437 | ~80 | ~65 | siamfcpp_googlenet-new.yaml | siamfcpp-googlenet-vot-md5_e14e9b6c82799602d777fd21a081c907.pkl |
| GoogLeNet | SiamFCppMultiTempTracker | VOT2018 | 0.587 | 0.150 |  0.467 | ~50 | ~45 | siamfcpp_googlenet-multi_temp.yaml | siamfcpp-googlenet-vot-md5_f2680ba074213ee39d82fcb84533a1a6.pkl |

__Nota__:

Points reported here are reproducible with PyTorch<=1.2.0. For PyTorch>=1.3.0, the reproducibility is not guaranteed due to a "breaking change" of PyTorch. See "Breaking Changes" under [release 1.3.0](https://github.com/pytorch/pytorch/releases) for detail.

However, we still recommend using the newest version of PyTorch as earlier versions usually carry numerous historical bugs (e.g. bugs with dataloader, ddp, etc.).

### GOT-10k

GOT-10k test configuration directory_experiments/siamfcpp/test/got10k_

| Backbone | Pipeline | Dataset | AO | SR.50 | SR.75 | Config. Filename | Model Filename |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AlexNet | SiamFCppTracker | GOT-10k-val | 72.0 | 85.0 | 63.3 | siamfcpp_alexnet_got.yaml | siamfcpp-alexnet-got-md5_5e01cf6271ad42e935032b61b05854d3.pkl|
| AlexNet | SiamFCppTracker | GOT-10k-test | 52.6 | 62.5 | 34.7 | siamfcpp_alexnet_got.yaml | siamfcpp-alexnet-got-md5_5e01cf6271ad42e935032b61b05854d3.pkl|
| GoogLeNet | SiamFCppTracker | GOT-10k-val | 76.4 | 90.4 | 71.8 | siamfcpp_googlenet_got.yaml | siamfcpp-googlenet-got-md5_e182dc4c3823427022eccf7313d740a7.pkl |
| GoogLeNet | SiamFCppTracker | GOT-10k-test | 60.7 | 73.7 | 46.4 | siamfcpp_googlenet_got.yaml | siamfcpp-googlenet-got-md5_e182dc4c3823427022eccf7313d740a7.pkl |
| ShuffleNetV2x0.5 | SiamFCppTracker | GOT-10k-val | 74.2 | 87.0| 67.1 | siamfcpp_shufflenetv2x0_5_got.yaml | siamfcpp-shufflenetv2x0_5-got-md5_d710ce17736d31a28bfe37cfbb997c5a.pkl |
| ShuffleNetV2x0.5 | SiamFCppTracker | GOT-10k-test | 52.9 | 61.7 | 38.1 | siamfcpp_shufflenetv2x0_5_got.yaml | siamfcpp-shufflenetv2x0_5-got-md5_d710ce17736d31a28bfe37cfbb997c5a.pkl |
| ShuffleNetV2x1.0 | SiamFCppTracker | GOT-10k-val | 76.6 | 88.8 | 71.5 | siamfcpp_shufflenetv2x1_0_got.yaml | siamfcpp-shufflenetv2x1_0-got-md5_aa824cc413b100bcb10f57c4d0e52423.pkl |
| ShuffleNetV2x1.0 | SiamFCppTracker | GOT-10k-test | 57.9 | 68.1 | 43.6 | siamfcpp_shufflenetv2x1_0_got.yaml | siamfcpp-shufflenetv2x1_0-got-md5_aa824cc413b100bcb10f57c4d0e52423.pkl |


### LaSOT

| Backbone | Pipeline | Dataset | Success | Precision | Normalized Precision | Config. Filename | Model Filename |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GoogLeNet | SiamFCppTracker | LaSOT-test | 55.7 | 55.6 | 58.9 | siamfcpp_googlenet-lasot.yaml | siamfcpp-googlenet-lasot-md5sum_434540569e163188d2bf47438e075529.pkl |

### TrackingNet

| Backbone | Pipeline | Dataset | Success | Precision | Normalized Precision | Config. Filename | Model Filename |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GoogLeNet | SiamFCppTracker | TrackingNet-TEST | 75.3 | 69.5 | 80.9 | siamfcpp_googlenet-trackingnet.yaml | siamfcpp-googlenet-vot_retrain-md5_0b2ab436b1b6866daad8f7915c135482.pkl |


### OTB-2015

| Backbone | Pipeline | Dataset | Success | Precision | Config. Filename | Model Filename |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AlexNet | SiamFCppTracker | OTB2015 | 68.0 | 88.4 | siamfcpp_googlenet-lasot.yaml | siamfcpp-googlenet-lasot-md5sum_434540569e163188d2bf47438e075529.pkl |
| GoogLeNet | SiamFCppTracker | OTB2015 | 68.2 | 89.6 | siamfcpp_googlenet-lasot.yaml | siamfcpp-googlenet-lasot-md5sum_434540569e163188d2bf47438e075529.pkl |

#### Pipeline

* SiamFCppTracker
  * [videoanalyst/pipeline/tracker/tracker_impl/siamfcpp_track.py](../videoanalyst/pipeline/tracker/tracker_impl/siamfcpp_track.py)
* SiamFCppMultiTempTracker
  * [videoanalyst/pipeline/tracker/tracker_impl/siamfcpp_track_multi_temp.py](../videoanalyst/pipeline/tracker/tracker_impl/siamfcpp_track_multi_temp.py)

### Remarks

* The results reported in our paper were produced by the implement under the internal deep learning framework. Afterwards, we reimplement our tracking method under PyTorch and there could be some differences between the reported results (under internal framework) and the real results (under PyTorch).
* Differences in hardware configuration (e.g. CPU style / GPU style) may influence some indexes (e.g. FPS)
  * Raw results here have been produced on a shared computing node equipped with _Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz_ and _Nvidia GeForce RTX 2080Ti_ .
  * "~" in the colomns for FPS denotes approximate values. FPS may vary due to factors other than code (e.g. hardware configuration / running status of machine).
* For VOT benchmark, models have been trained on ILSVRC-VID/DET, YoutubeBB, COCO, LaSOT, and GOT-10k (as described in our paper).

## Reproducibility

We have already observed several issues that are related to the reproducibility of the results under VOT benchmark. For example, under pytorch==1.1.0/1.2.0, the results of siamfcpp-googlenet are correct while under pytorch==1.3.0/1.4.0 not.

Following issues would influence the reproducibility of the results of existing models on VOT benchmark:

* PyTorch version
  * e.g. Type Promotion between 1.2.0 and 1.3.0, see Type Promotion on [PyTorch release notes](https://github.com/pytorch/pytorch/releases).
* CUDA/CUDNN version
  * 10.0 / 10.1
  * should be matched with the PyTorch (rebuilding may be needed)
* OpenCV version
  * Slight performance drop has been observed with the following change: 3.2.0.6 -> 4.1.0.25

We recommend keeping up-to-date with latest package version, and thus the points reported here counld be slightly away from the real points. Feel free to point them out in Issues if it is the case so that we can correct them.

Nevertheless, reproducibility of training under GOT-10k has been confirmed with repetition. Thus, there are no need to change software version (package/CUDA/CUDNN) unless you are obligated to verify the VOT result.

In addition, we strongly recommend to train and benchmark trackers on datasets like [GOT-10k](http://got-10k.aitestunion.com), not only because of its rigurous split of train/val/test, but also due to its large scale and diversity which make results stable.
