# -*- coding: utf-8 -*

import numpy as np

import torch


def imarray_to_tensor(arr):
    r"""
    Transpose & convert from numpy.array to torch.Tensor
    :param arr: numpy.array, (H, W, C)
    :return: torch.Tensor, (1, C, H, W)
    """
    arr = np.ascontiguousarray(
        arr.transpose(2, 0, 1)[np.newaxis, ...], np.float32)
    return torch.tensor(arr).type(torch.Tensor)


def tensor_to_imarray(t):
    r"""
    Perform naive detach / cpu / numpy process and then transpose
    :param t: torch.Tensor, (1, C, H, W)
    :return: numpy.array, (H, W, C)
    """
    arr = t.detach().cpu().numpy()
    return arr[0].transpose(1, 2, 0)


def tensor_to_numpy(t):
    r"""
    Perform naive detach / cpu / numpy process.
    :param t: torch.Tensor, (N, C, H, W)
    :return: numpy.array, (N, C, H, W)
    """
    arr = t.detach().cpu().numpy()
    return arr
