# -*- coding: utf-8 -*
from typing import Dict

import torch

def move_data_to_device(data_dict: Dict, dev: torch.device):
    for k in data_dict:
        data_dict[k] = data_dict[k].to(torch.device)

    return data_dict
