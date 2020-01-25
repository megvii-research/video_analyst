from paths import ROOT_PATH  # isort:skip

import os.path as osp
import cv2

from yacs.config import CfgNode
from videoanalyst.data.builder import build as build_dataloader
from videoanalyst.data.utils.misc import index_data
from videoanalyst.data.utils.visualization import show_img_FCOS

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.utils.misc import Timer

exp_cfg_path = osp.join(ROOT_PATH, "experiments/siamfcpp/train/siamfcpp_alexnet-trn.yaml")

cfg = CfgNode()

with open(exp_cfg_path) as f:
    cfg = CfgNode.load_cfg(f)

task = "track"


dataloader = build_dataloader(task, cfg.train[task].data)

# datasets = build_dataset(task, cfg.train[task].data.dataset)


num_minimatches = 1
with Timer(info="Dataloader:"):
    for ith in range(num_minimatches):
        training_data = next(dataloader)
        print("Batch", ith)
        for idx in range(training_data[list(training_data.keys())[0]].shape[0]):
            show_data = index_data(training_data, idx)
            cfg_viz = cfg.train[task].data.target[cfg.train[task].data.target.name]
            show_img_FCOS(cfg_viz, show_data)

# cv2.imshow("im_z", im_z)
# cv2.imshow("im_x", im_x)
# cv2.waitKey(0)

from IPython import embed;embed()
