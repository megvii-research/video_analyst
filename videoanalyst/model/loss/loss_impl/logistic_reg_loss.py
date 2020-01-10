
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from videoanalyst.model.loss.loss_base import TRACK_LOSSES

from .utils import SafeLog

eps = np.finfo(np.float32).tiny


class SigmoidCrossEntropyCenterness(nn.Module):

    def __init__(self, background=0, ignore_label=-1):
        super().__init__()
        self.background = background
        self.ignore_label = ignore_label
        self.safelog = SafeLog()
        self.register_buffer("t_one", torch.tensor(1., requires_grad=False))

    def forward(self, pred, label):
        """
        Center-ness loss
        :param pred: (B, HW), center-ness logits (BEFORE Sigmoid)
        :param label: (B, HW)
        :param cls_gt: (B, HW, C)
        :return:
        """
        mask = (1 - (label==self.background)).type(torch.Tensor).to(pred.device)
        not_neg_mask = (pred >= 0).type(torch.Tensor).to(pred.device)
        loss = (pred * not_neg_mask -
                pred * label +
                self.safelog(1. + torch.exp(-torch.abs(pred)))) * mask
        loss_residual = (-label*self.safelog(label)-(1-label)*self.safelog(1-label)) * mask # suppress loss residual (original vers.)
        loss = loss - loss_residual.detach()

        return loss.sum() / torch.max(mask.sum(), self.t_one)



if __name__=='__main__':
    B = 16
    HW = 17*17
    pred_cls = pred_ctr = torch.tensor(np.random.rand(B, HW, 1).astype(np.float32))
    pred_reg = torch.tensor(np.random.rand(B, HW, 4).astype(np.float32))

    gt_cls = torch.tensor(np.random.randint(2, size=(B, HW, 1)), dtype=torch.int8)
    gt_ctr = torch.tensor(np.random.rand(B, HW, 1).astype(np.float32))
    gt_reg = torch.tensor(np.random.rand(B, HW, 4).astype(np.float32))

    criterion_cls = SigmoidCrossEntropyRetina()
    loss_cls = criterion_cls(pred_cls, gt_cls)

    criterion_ctr = SigmoidCrossEntropyCenterness()
    loss_ctr = criterion_ctr(pred_ctr, gt_ctr, gt_cls)

    criterion_reg = IOULoss()
    loss_reg = criterion_reg(pred_reg, gt_reg, gt_cls)


    from IPython import embed;embed()
