
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .utils import SafeLog

from videoanalyst.model.loss.loss_base import TRACK_LOSSES

eps = np.finfo(np.float32).tiny

class SigmoidCrossEntropyRetina(nn.Module):

    def __init__(self, alpha=0.5, gamma=0, background_label=0, ignore_label=-1):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(float(alpha), requires_grad=False))
        self.register_buffer("gamma", torch.tensor(float(gamma), requires_grad=False))
        self.ignore_label = ignore_label
        self.background_label = background_label
        self.safelog = SafeLog()
        self.register_buffer("t_one", torch.tensor(1., requires_grad=False))


    def forward(self, pred, label):
        """
        Focal loss
        :param pred: shape=(B, HW, C), classification logits (BEFORE Sigmoid)
        :param label: shape=(B, HW)
        """
        mask = 1 - (label == self.ignore_label)
        mask = mask.type(torch.Tensor).to(label.device)
        vlabel = label * mask
        zero_mat = torch.zeros(pred.shape[0], pred.shape[1], pred.shape[2]+1)

        one_mat = torch.ones(pred.shape[0], pred.shape[1], pred.shape[2]+1)
        index_mat = vlabel.type(torch.LongTensor)

        onehot_ = zero_mat.scatter(2, index_mat, one_mat)
        onehot = onehot_[:, :, 1:].type(torch.Tensor).to(pred.device)

        pred = torch.sigmoid(pred)
        pos_part = (1 - pred)**self.gamma * onehot * self.safelog(pred)
        neg_part = pred**self.gamma * (1 - onehot) * self.safelog(1 - pred)
        loss = -(self.alpha * pos_part + (1 - self.alpha) * neg_part).sum(dim=2) * mask.squeeze(2)

        positive_mask = (label > 0).type(torch.Tensor).to(pred.device)

        return loss.sum() / torch.max(positive_mask.sum(), self.t_one)


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
