# -*- coding: utf-8 -*
import logging
import os.path as osp
from typing import Dict
import gc

from yacs.config import CfgNode

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from videoanalyst.utils import Timer, ensure_dir

from . import _DATA_LOGGER_NAME
from .adaptor_dataset import AdaptorDataset
from .datapipeline import builder as datapipeline_builder
from .sampler import builder as sampler_builder
from .target import builder as target_builder
from .transformer import builder as transformer_builder

logger = logging.getLogger("global")


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
        # build dummy dataset for purpose of dataset setup (e.g. caching path list)  
        logger.info("Build dummy AdaptorDataset")
        dummy_py_dataset = AdaptorDataset(dict(task=task, cfg=cfg),
                                          num_epochs=cfg.num_epochs,
                                          nr_image_per_epoch=cfg.nr_image_per_epoch)
        dummy_py_dataset.max_iter_per_epoch = cfg.nr_image_per_epoch // cfg.minibatch
        logger.info("Read dummy training sample")
        training_sample = dummy_py_dataset[0]  # read dummy sample
        del dummy_py_dataset
        gc.collect(generation=2)
        logger.info("Dummy AdaptorDataset destroyed.")
        # build real dataset
        logger.info("Build real AdaptorDataset")
        py_dataset = AdaptorDataset(dict(task=task, cfg=cfg),
                                    num_epochs=cfg.num_epochs,
                                    nr_image_per_epoch=cfg.nr_image_per_epoch)
        py_dataset.max_iter_per_epoch = cfg.nr_image_per_epoch // cfg.minibatch
        dataloader = DataLoader(
            py_dataset,
            batch_size=cfg.minibatch,
            shuffle=False,
            pin_memory=True,
            num_workers=cfg.num_workers,
            drop_last=True,
        )
    return dataloader


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
        cfg["exp_name"] = ""
        cfg["exp_save"] = "snapshots"
        cfg["num_epochs"] = 1
        cfg["minibatch"] = 32
        cfg["num_workers"] = 4
        cfg["nr_image_per_epoch"] = 150000
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
