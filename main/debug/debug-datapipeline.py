from paths import ROOT_PATH  # isort:skip

import os.path as osp
import cv2

from yacs.config import CfgNode
from videoanalyst.data.dataset.builder import build as build_dataset
from videoanalyst.data.sampler.builder import build as build_sampler
from videoanalyst.data.transformer.builder import build as build_transformer
from videoanalyst.data.target.builder import build as build_target

from videoanalyst.config.config import cfg as root_cfg

exp_cfg_path = osp.join(ROOT_PATH, "experiments/siamfcpp/train/siamfcpp_alexnet-trn.yaml")

cfg = CfgNode()

with open(exp_cfg_path) as f:
    cfg = CfgNode.load_cfg(f)

task = "track"

# datasets = build_dataset(task, cfg.train.data.dataset)
sampler =  build_sampler(task, cfg.train.data.sampler)
transformers = build_transformer(task, cfg.train.data.transformer)
transformer = transformers[0]
target = build_target(task, cfg.train.data.target)

t_start = cv2.getTickCount()
sampled_data = next(sampler)
transformed_data = transformer(sampled_data)
training_data = target(transformed_data)
im_z = transformed_data["data1"]["image"]
im_x = transformed_data["data2"]["image"]

t_elapsed = (cv2.getTickCount() - t_start)/cv2.getTickFrequency()
print("Elapsed:", t_elapsed)

# cv2.imshow("im_z", im_z)
# cv2.imshow("im_x", im_x)
# cv2.waitKey(0)

from IPython import embed;embed()
