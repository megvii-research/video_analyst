# MODEL_ZOO

## Download links

Models & Raw results:

* [Google Drive](https://drive.google.com/open?id=1UXshq4k9WKx4hNkdpOagJLXPR57ZkBkg)
* [baidu yun](https://pan.baidu.com/s/1uZ26iZyVJm50dJ3GoLCQ9w), code: rcsn

## Models

### Davis2017 val

VOS test configuration directory: _experiments/sat/test/

| MODEL | Pipeline | Dataset | J&F-Mean | J-Mean | J-Recall| J-Decay| F-Mean | F-Recall| F-Decaly | FPS@GTX2080Ti |Config. Filename|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SAT-Res50-transfer-param[1]|StateAwareTracker| DAVIS2017_val | 0.699069  |0.667666  |0.773786 | 0.174652 | 0.730471  | 0.82405 | 0.199258|~35|sat_res50-davis17.yaml 

__Nota__:

[1] Because the parameters of model are transferred from our internel training framework, the result is lower than what presented in the paper. We are working on rewrite the training process with pytorch. Please keep tuned for better models.
