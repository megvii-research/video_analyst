# -*- coding: utf-8 -*

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(precision=8)
from collections import OrderedDict
import numpy as np

from videoanalyst.model.common_opr.common_block import conv_bn_relu
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_head.taskhead_base import TRACK_HEADS


def get_xy_ctr(score_size, score_offset, total_stride):
    batch, fm_height, fm_width = 1, score_size, score_size

    y_list = torch.linspace(0., fm_height - 1., fm_height).reshape(1, fm_height, 1, 1).repeat(
        1, 1, fm_width, 1)  # .broadcast([1, fm_height, fm_width, 1])
    x_list = torch.linspace(0., fm_width - 1., fm_width).reshape(1, 1, fm_width, 1).repeat(
        1, fm_height, 1, 1)  # .broadcast([1, fm_height, fm_width, 1])
    xy_list = score_offset + torch.cat([x_list, y_list], 3) * total_stride
    xy_ctr = xy_list.repeat(batch, 1, 1, 1).reshape(
        batch, -1, 2)  # .broadcast([batch, fm_height, fm_width, 2]).reshape(batch, -1, 2)
    xy_ctr = xy_ctr.type(torch.Tensor)
    return xy_ctr


def get_box(xy_ctr, offsets):
    offsets = offsets.permute(0, 2, 3, 1)  # (B, H, W, C), C=4
    offsets = offsets.reshape(offsets.shape[0], -1, 4)

    # x0 = (xy_ctr[:, :, 0] - offsets[:, :, 0]).unsqueeze(2)
    # y0 = (xy_ctr[:, :, 1] - offsets[:, :, 1]).unsqueeze(2)
    # x1 = (xy_ctr[:, :, 0] + offsets[:, :, 2]).unsqueeze(2)
    # y1 = (xy_ctr[:, :, 1] + offsets[:, :, 3]).unsqueeze(2)
    # bboxes_pred = torch.cat([x0, y0, x1, y1], 2)

    xy0 = (xy_ctr[:, :, :] - offsets[:, :, :2])
    xy1 = (xy_ctr[:, :, :] + offsets[:, :, 2:])
    bboxes_pred = torch.cat([xy0, xy1], 2)

    return bboxes_pred


@TRACK_HEADS.register
class DenseboxHead(ModuleBase):
    default_hyper_params = {
        "total_stride": 8,
        "score_size": 17,
        "score_offset": 43,
    }

    def __init__(self):
        super(DenseboxHead, self).__init__()
        self.cls_p5_conv1 = conv_bn_relu(256, 256, stride=1, kszie=3, pad=0, has_bn=False)
        self.cls_p5_conv2 = conv_bn_relu(256, 256, stride=1, kszie=3, pad=0, has_bn=False)
        self.cls_p5_conv3 = conv_bn_relu(256, 256, stride=1, kszie=3, pad=0)

        self.cls_score_p5 = conv_bn_relu(256, 1, stride=1, kszie=1, pad=0, has_relu=False)
        self.ctr_score_p5 = conv_bn_relu(256, 1, stride=1, kszie=1, pad=0, has_relu=False)

        self.bbox_p5_conv1 = conv_bn_relu(256, 256, stride=1, kszie=3, pad=0, has_bn=False)
        self.bbox_p5_conv2 = conv_bn_relu(256, 256, stride=1, kszie=3, pad=0, has_bn=False)
        self.bbox_p5_conv3 = conv_bn_relu(256, 256, stride=1, kszie=3, pad=0)

        self.bbox_offsets_p5 = conv_bn_relu(256, 4, stride=1, kszie=1, pad=0, has_relu=False)

        self.bi = torch.nn.Parameter(torch.tensor(0.).type(torch.Tensor))
        self.si = torch.nn.Parameter(torch.tensor(1.).type(torch.Tensor))

        # initialze head
        conv_list = [
            self.cls_p5_conv1.conv, self.cls_p5_conv2.conv, self.cls_score_p5.conv,
            self.ctr_score_p5.conv, self.bbox_p5_conv1.conv, self.bbox_p5_conv2.conv,
            self.bbox_offsets_p5.conv
        ]
        conv_classifier = [self.cls_score_p5.conv]
        assert all(elem in conv_list for elem in conv_classifier)

        pi = 0.01
        bv = -np.log((1 - pi) / pi)
        for ith in range(len(conv_list)):
            # fetch conv from list
            conv = conv_list[ith]
            # torch.nn.init.normal_(conv.weight, std=0.01) # from megdl impl.
            torch.nn.init.normal_(conv.weight, std=0.001)  #0.0001)
            # nn.init.kaiming_uniform_(conv.weight, a=np.sqrt(5))  # from PyTorch default implementation
            # nn.init.kaiming_uniform_(conv.weight, a=0)  # from PyTorch default implementation
            if conv in conv_classifier:
                torch.nn.init.constant_(conv.bias, torch.tensor(bv))
            else:
                # torch.nn.init.constant_(conv.bias, 0)
                # from PyTorch default implementation
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(conv.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(conv.bias, -bound, bound)

    def forward(self, c_out, r_out):
        # classification head
        cls = self.cls_p5_conv1(c_out)
        cls = self.cls_p5_conv2(cls)
        cls = self.cls_p5_conv3(cls)  #todo
        # classification score
        cls_score = self.cls_score_p5(cls)  #todo
        cls_score = cls_score.permute(0, 2, 3, 1)
        cls_score = cls_score.reshape(cls_score.shape[0], -1, 1)
        # center-ness score
        ctr_score = self.ctr_score_p5(cls)  #todo
        ctr_score = ctr_score.permute(0, 2, 3, 1)
        ctr_score = ctr_score.reshape(ctr_score.shape[0], -1, 1)
        # regression head
        bbox = self.bbox_p5_conv1(r_out)
        bbox = self.bbox_p5_conv2(bbox)
        bbox = self.bbox_p5_conv3(bbox)
        offsets = self.bbox_offsets_p5(bbox)
        offsets = torch.exp(self.si * offsets + self.bi) * self.total_stride
        # bbox decoding
        self.fm_ctr = self.fm_ctr.to(offsets.device)
        fcos_bbox = get_box(self.fm_ctr, offsets)

        return [cls_score, ctr_score, fcos_bbox]

    def update_params(self):
        self.score_size = self._hyper_params["score_size"]
        self.score_offset = self._hyper_params["score_offset"]
        self.total_stride = self._hyper_params["total_stride"]
        ctr = get_xy_ctr(self.score_size, self.score_offset, self.total_stride)
        self.fm_ctr = ctr
        self.fm_ctr.require_grad = False
