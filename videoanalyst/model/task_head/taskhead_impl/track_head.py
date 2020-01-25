# -*- coding: utf-8 -*

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from videoanalyst.model.common_opr.common_block import conv_bn_relu
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_head.taskhead_base import TRACK_HEADS

torch.set_printoptions(precision=8)


def get_xy_ctr(score_size, score_offset, total_stride):
    batch, fm_height, fm_width = 1, score_size, score_size

    y_list = torch.linspace(0., fm_height - 1., fm_height).reshape(
        1, fm_height, 1, 1).repeat(1, 1, fm_width,
                                   1)  # .broadcast([1, fm_height, fm_width, 1])
    x_list = torch.linspace(0., fm_width - 1., fm_width).reshape(
        1, 1, fm_width, 1).repeat(1, fm_height, 1,
                                  1)  # .broadcast([1, fm_height, fm_width, 1])
    xy_list = score_offset + torch.cat([x_list, y_list], 3) * total_stride
    xy_ctr = xy_list.repeat(batch, 1, 1, 1).reshape(
        batch, -1,
        2)  # .broadcast([batch, fm_height, fm_width, 2]).reshape(batch, -1, 2)
    xy_ctr = xy_ctr.type(torch.Tensor)
    return xy_ctr


def get_box(xy_ctr, offsets):
    offsets = offsets.permute(0, 2, 3, 1)  # (B, H, W, C), C=4
    offsets = offsets.reshape(offsets.shape[0], -1, 4)

    xy0 = (xy_ctr[:, :, :] - offsets[:, :, :2])
    xy1 = (xy_ctr[:, :, :] + offsets[:, :, 2:])
    bboxes_pred = torch.cat([xy0, xy1], 2)

    return bboxes_pred


@TRACK_HEADS.register
class DenseboxHead(ModuleBase):
    r"""
    Densebox Head for siamfcpp

    Hyper-parameter
    ---------------
    total_stride: int
        stride in backbone
    score_size: int
        final feature map
    x_size: int
        search image size
    num_conv3x3: int
        number of conv3x3 tiled in head
    head_conv_bn: list
        has_bn flag of conv3x3 in head, list with length of num_conv3x3
    head_width: int
        feature width in head structure
    """
    default_hyper_params = dict(
        total_stride=8,
        score_size=17,
        x_size=303,
        num_conv3x3=3,
        head_conv_bn=[False, False, True],
        head_width=256,
    )

    def __init__(self):
        super(DenseboxHead, self).__init__()

        self.bi = torch.nn.Parameter(torch.tensor(0.).type(torch.Tensor))
        self.si = torch.nn.Parameter(torch.tensor(1.).type(torch.Tensor))

        self.cls_convs = []
        self.bbox_convs = []

    def forward(self, c_out, r_out):
        # classification head
        num_conv3x3 = self._hyper_params['num_conv3x3']
        cls = c_out
        bbox = r_out

        for i in range(0, num_conv3x3):
            # cls = self.cls_conv3x3_list[i](cls)
            # bbox = self.bbox_conv3x3_list[i](bbox)
            cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
            bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)

        # classification score
        cls_score = self.cls_score_p5(cls)  #todo
        cls_score = cls_score.permute(0, 2, 3, 1)
        cls_score = cls_score.reshape(cls_score.shape[0], -1, 1)
        # center-ness score
        ctr_score = self.ctr_score_p5(cls)  #todo
        ctr_score = ctr_score.permute(0, 2, 3, 1)
        ctr_score = ctr_score.reshape(ctr_score.shape[0], -1, 1)
        # regression
        offsets = self.bbox_offsets_p5(bbox)
        offsets = torch.exp(self.si * offsets + self.bi) * self.total_stride
        # bbox decoding
        self.fm_ctr = self.fm_ctr.to(offsets.device)
        bbox = get_box(self.fm_ctr, offsets)

        return [cls_score, ctr_score, bbox]

    def update_params(self):
        x_size = self._hyper_params["x_size"]
        score_size = self._hyper_params["score_size"]
        total_stride = self._hyper_params["total_stride"]
        score_offset = (x_size - 1 - (score_size - 1) * total_stride) // 2
        self._hyper_params["score_offset"] = score_offset

        self.score_size = self._hyper_params["score_size"]
        self.total_stride = self._hyper_params["total_stride"]
        self.score_offset = self._hyper_params["score_offset"]
        ctr = get_xy_ctr(self.score_size, self.score_offset, self.total_stride)
        self.fm_ctr = ctr
        self.fm_ctr.require_grad = False

        self._make_conv3x3()
        self._make_conv_output()

    def _make_conv3x3(self):
        num_conv3x3 = self._hyper_params['num_conv3x3']
        head_conv_bn = self._hyper_params['head_conv_bn']
        head_width = self._hyper_params['head_width']
        self.cls_conv3x3_list = []
        self.bbox_conv3x3_list = []
        for i in range(num_conv3x3):
            # is_last_conv = (i >= num_conv3x3)
            cls_conv3x3 = conv_bn_relu(head_width,
                                       head_width,
                                       stride=1,
                                       kszie=3,
                                       pad=0,
                                       has_bn=head_conv_bn[i])

            bbox_conv3x3 = conv_bn_relu(head_width,
                                        head_width,
                                        stride=1,
                                        kszie=3,
                                        pad=0,
                                        has_bn=head_conv_bn[i])
            setattr(self, 'cls_p5_conv%d' % (i + 1), cls_conv3x3)
            setattr(self, 'bbox_p5_conv%d' % (i + 1), bbox_conv3x3)
            self.cls_conv3x3_list.append(cls_conv3x3)
            self.bbox_conv3x3_list.append(bbox_conv3x3)

    def _make_conv_output(self):
        head_width = self._hyper_params['head_width']
        self.cls_score_p5 = conv_bn_relu(head_width,
                                         1,
                                         stride=1,
                                         kszie=1,
                                         pad=0,
                                         has_relu=False)
        self.ctr_score_p5 = conv_bn_relu(head_width,
                                         1,
                                         stride=1,
                                         kszie=1,
                                         pad=0,
                                         has_relu=False)
        self.bbox_offsets_p5 = conv_bn_relu(head_width,
                                            4,
                                            stride=1,
                                            kszie=1,
                                            pad=0,
                                            has_relu=False)
