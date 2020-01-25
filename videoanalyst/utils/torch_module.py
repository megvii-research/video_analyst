# -*- coding: utf-8 -*
from typing import Dict

import torch
from torch import nn


def move_data_to_device(data_dict: Dict, dev: torch.device):
    for k in data_dict:
        data_dict[k] = data_dict[k].to(dev)

    return data_dict


def unwrap_model(model):
    r""" unwrap nn.dataparallel wrapped module for model serialization """
    return model.module if isinstance(model, nn.DataParallel) else model

def convert_data_to_dtype(data_dict: Dict[str, torch.Tensor], dtype: torch.dtype = torch.Tensor):
    r"""
    Convert

    Parameters
    ----------
    data_dict: Dict[str, torch.Tensor]
        data dict to convert
    dtype: torch.dtype
        target dtype, to be passed to torch.Tensor.astype(dtype)

    Returns
    -------
    data_dict
        converted data dict
    """
    for k in data_dict:
        data_dict[k] = data_dict[k].type(dtype)

    return data_dict
