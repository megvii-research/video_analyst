## Download links
Models & Raw results:
* [Google Drive](https://drive.google.com/open?id=1XhWIU1KIt9wvFpzZqEDaX-GrgZ9AVcOC)
* [Baidu Disk](https://pan.baidu.com/s/19GhRrv2RcEQBFAJ-TNs8mg), code: qvfq

## Models
| Backbone | Pipeline | Dataset | A | R | EAO | FPS | Config. Filename | Model filename |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AlexNet | Single template | VOT2018 |0.588 | 0.243 | 0.373| ~200 | siamfcpp_alexnet.yaml | siamfcpp-alexnet-vot-md5_18fd31a2f94b0296c08fff9b0f9ad240.pkl|
| AlexNet | Simple multi-template strategy| VOT2018 | 0.597 | 0.215 | 0.370 | ~90 | siamfcpp_alexnet-multi_temp.yaml | siamfcpp-alexnet-vot-md5_18fd31a2f94b0296c08fff9b0f9ad240.pkl|
| GoogLeNet | Single template | VOT2018 | 0.583 | 0.173 | 0.426 | ~80 | siamfcpp_googlenet.yaml | siamfcpp-googlenet-vot-md5_f2680ba074213ee39d82fcb84533a1a6.pkl |
| GoogLeNet | Simple multi-template strategy | VOT2018 | 0.587 | 0.150 |  0.467 | ~60 | siamfcpp_googlenet-multi_temp.yaml | siamfcpp-googlenet-vot-md5_f2680ba074213ee39d82fcb84533a1a6.pkl |

#### Remarks
* The results reported in our paper were produced by the implement under the internal deep learning framework. Afterwards, we reimplement our tracking method under PyTorch and there could be some differences between the reported results (under internal framework) and the real results (under PyTorch).
* Differences in hardware configuration (e.g. CPU style / GPU style) may influence some indexes (e.g. FPS)
    * Results here have been produced on a shared computing node equipped with _Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz_ and _Nvidia GeForce RTX 2080Ti_ .
* For VOT benchmark, models have been trained on ILSVRC-VID/DET, YoutubeBB, COCO, LaSOT, and GOT-10k (as described in our paper).
