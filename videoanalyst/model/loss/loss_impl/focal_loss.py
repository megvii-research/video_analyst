# -*- coding: utf-8 -*

import torch

from ...common_opr.common_loss import sigmoid_focal_loss_jit
from ...module_base import ModuleBase
from ..loss_base import TRACK_LOSSES


@TRACK_LOSSES.register
class FocalLoss(ModuleBase):

    default_hyper_params = dict(
        name="focal_loss",
        background=0,
        ignore_label=-1,
        weight=1.0,
        alpha=0.5,
        gamma=0.0,
    )

    def __init__(self, ):
        super().__init__()
        self.register_buffer("t_one", torch.tensor(1., requires_grad=False))

    def forward(self, pred_data, target_data):
        r"""
        Focal loss
        :param pred: shape=(B, HW, C), classification logits (BEFORE Sigmoid)
        :param label: shape=(B, HW)
        """
        r"""
        Focal loss
        Arguments
        ---------
        pred: torch.Tensor
            classification logits (BEFORE Sigmoid)
            format: (B, HW)
        label: torch.Tensor
            training label
            format: (B, HW)

        Returns
        -------
        torch.Tensor
            scalar loss
            format: (,)
        """
        pred = pred_data["cls_pred"]
        label = target_data["cls_gt"]
        mask = ~(label == self._hyper_params["ignore_label"])
        mask = mask.type(torch.Tensor).to(label.device)
        vlabel = label * mask
        zero_mat = torch.zeros(pred.shape[0], pred.shape[1], pred.shape[2] + 1)

        one_mat = torch.ones(pred.shape[0], pred.shape[1], pred.shape[2] + 1)
        index_mat = vlabel.type(torch.LongTensor)
        onehot_ = zero_mat.scatter(2, index_mat, one_mat)
        onehot = onehot_[:, :, 1:].type(torch.Tensor).to(pred.device)
        loss = sigmoid_focal_loss_jit(pred, onehot, self._hyper_params["alpha"],
                                      self._hyper_params["gamma"], "none")
        positive_mask = (label > 0).type(torch.Tensor).to(pred.device)
        loss = (loss.sum(dim=2) * mask.squeeze(2)).sum() / (torch.max(
            positive_mask.sum(), self.t_one)) * self._hyper_params["weight"]
        extra = dict()
        return loss, extra
