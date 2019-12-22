# -*- coding: utf-8 -*

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_bn_relu(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 kszie=3,
                 pad=0,
                 has_bn=True,
                 has_relu=True,
                 bias=True):
        """
        Basic block with one conv, one bn, one relu in series.
        :param in_channel: number of input channels
        :param out_channel: number of output channels
        :param stride: stride number
        :param kszie: kernel size
        :param pad: padding on each edge
        :param has_bn: use bn or not
        :param has_relu: use relu or not
        :param bias: conv has bias or not
        """
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=kszie,
                              stride=stride,
                              padding=pad,
                              bias=bias)

        if has_bn:
            self.bn = nn.BatchNorm2d(out_channel)
        else:
            self.bn = None

        if has_relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def xcorr_depthwise(x, kernel):
    """
    Depthwise cross correlation. e.g. used for template matching in Siamese tracking network
    :param x:
    :param kernel: smaller than x
    :return:
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out
