from paths import ROOT_PATH  # isort:skip

import os.path as osp
import cv2

from yacs.config import CfgNode
from videoanalyst.data.builder import build as build_dataloader

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.utils.misc import Timer

exp_cfg_path = osp.join(ROOT_PATH, "experiments/siamfcpp/train/siamfcpp_alexnet-trn.yaml")

cfg = CfgNode()

with open(exp_cfg_path) as f:
    cfg = CfgNode.load_cfg(f)

task = "track"


dataloader = build_dataloader(task, cfg.train.data)

# datasets = build_dataset(task, cfg.train.data.dataset)


num_minimatches = 100
with Timer(info="Dataloader:"):
    for ith in range(num_minimatches):
        training_data = next(dataloader)
        print(ith)

# cv2.imshow("im_z", im_z)
# cv2.imshow("im_x", im_x)
# cv2.waitKey(0)


from IPython import embed;embed()
