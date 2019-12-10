# -*- coding: utf-8 -*
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from videoanalyst.model.loss.loss_base import TRACK_LOSSES
from videoanalyst.model.module_base import ModuleBase

eps = np.finfo(np.float32).tiny


class SafeLog(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("t_eps", torch.tensor(eps, requires_grad=False))

    def forward(self, t):
        return torch.log(torch.max(self.t_eps, t))


@TRACK_LOSSES.register
class IOULoss(ModuleBase):

    default_hyper_params = {"background": 0, "ignore_label": -1, "weight": 1.0}

    def __init__(self, background=0, ignore_label=-1):
        super().__init__()
        self.safelog = SafeLog()
        self.register_buffer("t_one", torch.tensor(1., requires_grad=False))
        self.register_buffer("t_zero", torch.tensor(0., requires_grad=False))

    def update_params(self):
        self.background = self._hyper_params["background"]
        self.ignore_label = self._hyper_params["ignore_label"]
        self.weight = self._hyper_params["weight"]

    def forward(self, pred, gt, cls_gt):
        mask = ((1 - (cls_gt == self.background)) *
                (1 - (cls_gt == self.ignore_label))).detach()
        mask = mask.type(torch.Tensor).squeeze(2).to(pred.device)

        aog = torch.abs(gt[:, :, 2] - gt[:, :, 0] +
                        1) * torch.abs(gt[:, :, 3] - gt[:, :, 1] + 1)
        aop = torch.abs(pred[:, :, 2] - pred[:, :, 0] +
                        1) * torch.abs(pred[:, :, 3] - pred[:, :, 1] + 1)

        iw = torch.min(pred[:, :, 2], gt[:, :, 2]) - torch.max(
            pred[:, :, 0], gt[:, :, 0]) + 1
        ih = torch.min(pred[:, :, 3], gt[:, :, 3]) - torch.max(
            pred[:, :, 1], gt[:, :, 1]) + 1
        inter = torch.max(iw, self.t_zero) * torch.max(ih, self.t_zero)

        union = aog + aop - inter
        iou = torch.max(inter / union, self.t_zero)
        loss = -self.safelog(iou)

        # from IPython import embed;embed()
        loss = (loss * mask).sum() / torch.max(mask.sum(),
                                               self.t_one) * self.weight
        iou = iou.detach()
        iou = (iou * mask).sum() / torch.max(mask.sum(), self.t_one)

        return loss, iou
