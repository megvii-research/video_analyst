# -*- coding: utf-8 -*
from typing import Dict

import torch
from torch import nn
import torch.distributed as dist


def move_data_to_device(data_dict: Dict, dev: torch.device):
    for k in data_dict:
        data_dict[k] = data_dict[k].to(dev)

    return data_dict


def unwrap_model(model):
    r""" unwrap nn.dataparallel wrapped module for model serialization """
    return model.module if isinstance(
        model,
        (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model


def convert_data_to_dtype(data_dict: Dict[str, torch.Tensor],
                          dtype: torch.dtype = torch.Tensor):
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


def average_gradients(model):
    r""" Gradient averaging. 
         from https://pytorch.org/tutorials/intermediate/dist_tuto.html
         to be called after _loss.backward()_ and before _optimizer.step()_
    """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size
