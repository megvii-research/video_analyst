# -*- coding: utf-8 -*
import logging
import os.path as osp
from typing import Dict

from yacs.config import CfgNode

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from videoanalyst.utils import ensure_dir, Timer

from . import _DATA_LOGGER_NAME
from .adaptor_dataset import AdaptorDataset, AdaptorIterableDataset
from .datapipeline import builder as datapipeline_builder
from .sampler import builder as sampler_builder
from .target import builder as target_builder
from .transformer import builder as transformer_builder

logger = logging.getLogger("global")
from itertools import chain
class MultiStreamDataloader:
    def __init__(self, datasets):
        self.datasets = datasets
    def __length__(self):
        return len(self.datasets[0])

    def get_stream_loaders(self):
        return zip(*[DataLoader(dataset, num_workers=1, batch_size=None, pin_memory=True) for dataset in self.datasets])
    
    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            with Timer(name="defalt_collert", logger=logger, verbose=True):
                data =  default_collate(list(chain(*batch_parts)))
            yield data

def build(task: str, cfg: CfgNode) -> DataLoader:
    r"""
    Arguments
    ---------
    task: str
        task name (track|vos)
    cfg: CfgNode
        node name: data
    """
    data_logger = build_data_logger(cfg)

    if task == "track":
        '''
        py_dataset = AdaptorDataset(dict(task=task, cfg=cfg),
                                    num_epochs=cfg.num_epochs,
                                    nr_image_per_epoch=cfg.nr_image_per_epoch)

        dataloader = DataLoader(py_dataset,
                                batch_size=cfg.minibatch,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=cfg.num_workers,
                                drop_last=True, 
                                )
        '''
        py_datasets = AdaptorIterableDataset.split_datasets(dict(task=task, cfg=cfg),
                                    num_epochs=cfg.num_epochs,
                                    nr_image_per_epoch=cfg.nr_image_per_epoch,
                                    batch_size=cfg.minibatch, max_workers=cfg.num_workers)
        dataloader = MultiStreamDataloader(py_datasets)
    return iter(dataloader)


def get_config() -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {"track": CfgNode(), "vos": CfgNode()}

    for task in cfg_dict:
        cfg = cfg_dict[task]

        module = AdaptorDataset
        # modify _AdaptorDataset.default_hyper_params_ to add new config name under _data_
        hps = module.default_hyper_params
        for hp_name in hps:
            cfg[hp_name] = hps[hp_name]

        cfg["datapipeline"] = datapipeline_builder.get_config()[task]
        cfg["sampler"] = sampler_builder.get_config()[task]
        cfg["transformer"] = transformer_builder.get_config()[task]
        cfg["target"] = target_builder.get_config()[task]

    return cfg_dict


def build_data_logger(cfg: CfgNode) -> logging.Logger:
    r"""Build logger for data module
    
    Parameters
    ----------
    cfg : CfgNode
        cfg, node name: data
    
    Returns
    -------
    logging.Logger
        logger built with file handler at "exp_save/exp_name/logs/data.log"
    """
    log_dir = osp.join(cfg.exp_save, cfg.exp_name, "logs")
    ensure_dir(log_dir)
    log_file = osp.join(log_dir, "data.log")
    data_logger = logging.getLogger(_DATA_LOGGER_NAME)
    data_logger.setLevel(logging.INFO)
    # file handler
    fh = logging.FileHandler(log_file)
    format_str = "[%(asctime)s - %(filename)s] - %(message)s"
    formatter = logging.Formatter(format_str)
    fh.setFormatter(formatter)
    # add file handler
    data_logger.addHandler(fh)
    logger.info("Data log file registered at: %s" % log_file)
    data_logger.info("Data logger built.")

    return data_logger
