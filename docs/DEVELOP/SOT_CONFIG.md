## Pipeline

Basic sot tracker pipelines:
* _videoanalyst/pipeline/tracker_impl/siamfcpp_track.py_

Basic API
* init(im, state)
* update(im[, rect])

P.S. state: bounding box (format: xywh)

## Config
e.g. _experiments/siamfcpp/test/got10k/siamfcpp_googlenet-got.yaml_
* model
  * task_head
    * DenseboxHead
      * total_stride: downsample ratio (8 for SOT)
      * score_size: size of final dense prediction size
      * x_size: search image input size
      * num_conv3x3: number of conv3x3 in head. Note that each conv3x3 brings a shrinkage of 2 pixel in _score_size_
      * head_conv_bn: List[bool], control BN config in head
* pipeline
  * SiamFCppTracker:
    * test_lr: control the bbox smoothing factor, larger test_lr -> less smoothing
    * window_influence: control the penalization on spatial motion    
    * penalty_k: control the penalization on bbox prediction change
    * x_size: keep as the same in _model_
    * num_conv3x3: keep as the same in _model_
