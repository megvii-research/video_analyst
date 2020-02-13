from paths import ROOT_PATH  # isort:skip

import os.path as osp

import cv2
import numpy as np
from yacs.config import CfgNode

import torch

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.model.loss.builder import build as build_loss
from videoanalyst.utils.misc import Timer

exp_cfg_path = osp.join(ROOT_PATH,
                        "experiments/siamfcpp/train/siamfcpp_alexnet-trn.yaml")

cfg = CfgNode()

with open(exp_cfg_path) as f:
    cfg = CfgNode.load_cfg(f)

task = "track"

losses = build_loss(task, cfg.train.model.losses)

if __name__ == '__main__':
    B = 16
    HW = 17 * 17
    pred_cls = pred_ctr = torch.tensor(
        np.random.rand(B, HW, 1).astype(np.float32))
    pred_reg = torch.tensor(np.random.rand(B, HW, 4).astype(np.float32))

    # gt_cls = torch.tensor(np.random.randint(2, size=(B, HW, 1)), dtype=torch.int8)
    gt_cls = torch.tensor(np.random.randint(2, size=(B, HW, 1)),
                          dtype=torch.float)
    gt_ctr = torch.tensor(np.random.rand(B, HW, 1).astype(np.float32))
    gt_reg = torch.tensor(np.random.rand(B, HW, 4).astype(np.float32))

    pred_data = dict(
        cls_pred=pred_cls,
        ctr_pred=pred_ctr,
        box_pred=pred_reg,
    )

    target_data = dict(
        cls_gt=gt_cls,
        ctr_gt=gt_ctr,
        box_gt=gt_reg,
    )

    criterion_cls, criterion_ctr, criterion_reg = losses

    loss_cls = criterion_cls(pred_data, target_data)

    loss_ctr = criterion_ctr(pred_data, target_data)

    loss_reg = criterion_reg(pred_data, target_data)

    print(loss_cls, loss_ctr)
    from IPython import embed
    embed()
