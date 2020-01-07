import numpy as np
import cv2

from yacs.config import CfgNode

from videoanalyst.evaluation.got_benchmark.datasets import got10k
from videoanalyst.data.dataset.dataset_base import TRACK_DATASETS, DatasetBase

@TRACK_DATASETS.register
class GOT10kDataset(DatasetBase):
    r"""
    GOT-10k dataset helper

    Hyper-parameters
    ----------------
    dataset_root: str
        path to root of the dataset
    subset: str
        dataset split name (train|val|test)
    """
    default_hyper_params = dict(
        dataset_root="datasets/GOT-10k",
        subset="train",
        ratio=1,
        max_diff=100,
    )
    def __init__(self, cfg: CfgNode) -> None:
        r"""
        Crete datset with config

        Arguments
        ---------
        cfg: CfgNode
            dataset config
        """
        super().__init__(cfg)
        self._state["datset"] = None

    def update_params(self):
        r"""
        an interface for update params
        """
        datset_root = self._hyper_params["datset_root"]
        subset = self._hyper_params["subset"]
        self._state["datset"] = got10k(datset_root, subset=subset)

    def __getitem__(self, item: int) -> dict:
        img_files, anno = self._state["datset"][item]
        sequence_data = dict(image=img_files, anno=anno)

        return sequence_data

    def __len__(self):
        return len(self._state["datset"])

