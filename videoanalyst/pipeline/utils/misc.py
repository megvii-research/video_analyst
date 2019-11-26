import numpy as np
import torch


def imarray_to_tensor(arr):
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1)[np.newaxis, ...], np.float32)
    return torch.tensor(arr).type(torch.Tensor)

def tensor_to_imarray(t):
    arr = t.detach().cpu().numpy()
    return arr[0].transpose(1, 2, 0)

def tensor_to_numpy(t):
    """
    Make
    Perform naive detach / cpu / numpy process.
    :param t: torch.Tensor
    :return: numpy.array
    """
    arr = t.detach().cpu().numpy()
    return arr
